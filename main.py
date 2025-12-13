import os
import io
import time
import random
import hashlib
from pathlib import Path

# UI
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, ListView, ListItem, Label, Button, Sparkline
from textual import on, work
from textual.binding import Binding
from textual.events import Click
from textual.reactive import reactive

# Audio & Data
import pygame
import numpy as np
from pydub import AudioSegment
from PIL import Image

# Metadata
from mutagen import File as MutagenFile
from mutagen.id3 import ID3 

# --- 路径配置 ---
MUSIC_DIR = Path("J:\音乐\Hitorie")
# ----------------

THEME_COLOR = "#00ff9d" 

class AudioEngine:
    def __init__(self):
        pygame.mixer.init(frequency=44100)
        self.raw_data = None
        self.frame_rate = 44100
        self.duration_sec = 0
        self.analyzing = False
        self.start_offset = 0
        self.current_file = None

    def load_and_play(self, path, start_time=0):
        self.analyzing = True
        self.start_offset = start_time
        self.current_file = path 
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play(start=start_time)
        except Exception as e:
            print(f"Play Error: {e}")

    def get_current_time(self):
        if pygame.mixer.music.get_busy():
            return self.start_offset + (pygame.mixer.music.get_pos() / 1000.0)
        return self.start_offset

    def background_analyze(self, path):
        try:
            f = MutagenFile(path)
            if f and hasattr(f, 'info'):
                self.duration_sec = f.info.length
            
            audio = AudioSegment.from_file(path)
            if len(audio) > 20000: audio = audio[:20000] 
            self.frame_rate = audio.frame_rate
            self.raw_data = np.array(audio.set_channels(1).get_array_of_samples())
        except Exception as e: pass
        finally: self.analyzing = False

    def get_spectrum(self, bars=60):
        if self.raw_data is None or len(self.raw_data) == 0:
            return [0]*bars
        
        current_abs = self.get_current_time()
        display_time = current_abs % 20 
        idx = int(display_time * self.frame_rate)
        chunk = 2048
        
        if idx + chunk > len(self.raw_data): return [0]*bars
        
        data = np.abs(self.raw_data[idx:idx+chunk])
        step = len(data) // bars
        if step == 0: return [0]*bars
        res = [int(np.mean(data[i*step:(i+1)*step])) for i in range(bars)]
        return [min(max(x // 100, 1), 8) for x in res]

class ManualProgressBar(Label):
    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.percentage = 0.0

    def update_progress(self, pct):
        self.percentage = max(0.0, min(pct, 1.0))
        self.refresh() 

    def render(self):
        width = self.content_size.width or 50
        filled_len = int(width * self.percentage)
        bar = f"[{THEME_COLOR}]" + ("━" * filled_len) + "[/]" + \
              f"[#ffffff]●[/]" + \
              f"[#333333]" + ("─" * (width - filled_len - 1)) + "[/]"
        return bar

    def on_click(self, event: Click):
        pct = event.x / self.size.width
        self.app.on_seek_request(pct)

class NeonPlayer(App):
    CSS = """
    Screen { layout: horizontal; background: #0f0f0f; }
    
    #sidebar { 
        width: 30; height: 100%; 
        background: #141414; 
        border-right: vkey #333333; 
        dock: left;
    }
    .list-header { 
        padding: 1; text-align: center; text-style: bold; 
        background: #1a1a1a; color: #888888; 
    }
    ListView { background: #141414; }
    ListItem { padding: 1 2; color: #aaaaaa; }
    ListItem:hover { background: #222222; }
    ListItem.-selected { background: #1a1a1a; color: #00ff9d; border-left: solid #00ff9d; }

    #main-view { width: 1fr; height: 100%; }

    #info-container {
        width: 100%; height: 1fr; 
        align: center middle; padding: 1;
        overflow: hidden; 
    }

    #album-art { 
        height: auto; max-height: 22; width: auto;
        border: heavy #333333; margin-bottom: 1; 
        text-align: center; background: #000000; color: #00ff9d;
    }

    #meta-title { text-style: bold; color: #ffffff; text-align: center; }
    #meta-artist { color: #00ff9d; text-align: center; margin-bottom: 1; }
    #meta-album { color: #666666; text-align: center; text-style: italic; display: none; }

    #bottom-pane {
        dock: bottom; height: auto;
        background: #0f0f0f; padding: 1 4;
        border-top: solid #222222;
    }

    #time-label { color: #555555; text-align: right; width: 100%; }
    ManualProgressBar { width: 100%; height: 1; margin: 0; }
    Sparkline { height: 3; width: 100%; margin: 1 0; color: #00ff9d; opacity: 60%; }
    
    #controls { layout: horizontal; align: center middle; height: 3; margin-top: 1; }
    Button {
        min-width: 14; height: 3; margin: 0 1; border: none;
        background: #222222; color: #eeeeee;
    }
    Button:hover { background: #00ff9d; color: #000000; }
    #btn-play { background: #2a2a2a; border: solid #00ff9d; }
    """

    BINDINGS = [
        Binding("space", "toggle_play", "播放/暂停"),
        Binding("q", "quit", "退出")
    ]

    current_metadata = reactive({"title": "Waiting...", "artist": "-", "album": "-"})

    def __init__(self):
        super().__init__()
        self.audio = AudioEngine()
        self.playlist = []
        self.current_idx = -1
        self.is_playing = False
        self.is_seeking = False

    def compose(self) -> ComposeResult:
        with Container(id="sidebar"):
            yield Label("P L A Y L I S T", classes="list-header")
            yield ListView(id="track-list")
        
        with Container(id="main-view"):
            with Vertical(id="info-container"):
                yield Label("", id="album-art")
                yield Label("No Track Selected", id="meta-title")
                yield Label("-", id="meta-artist")
                yield Label("-", id="meta-album")

            with Vertical(id="bottom-pane"):
                yield Label("00:00 / 00:00", id="time-label")
                yield ManualProgressBar(id="progress")
                yield Sparkline(summary_function=max, id="spectrum")
                with Container(id="controls"):
                    yield Button("PREV", id="btn-prev")
                    yield Button("PLAY", id="btn-play")
                    yield Button("NEXT", id="btn-next")

    def on_mount(self):
        self.load_files()
        self.set_interval(0.1, self.update_ui_tick)

    def load_files(self):
        lv = self.query_one("#track-list", ListView)
        if MUSIC_DIR.exists():
            files = [p for p in MUSIC_DIR.glob("*") if p.suffix.lower() in ['.mp3', '.flac', '.wav']]
            files.sort(key=lambda x: x.name)
            for f in files:
                self.playlist.append(f)
                lv.append(ListItem(Label(f" {f.name}")))

    def update_ui_tick(self):
        data = self.audio.get_spectrum(bars=90)
        self.query_one("#spectrum", Sparkline).data = data

        if not self.is_playing or self.is_seeking: return
        
        current = self.audio.get_current_time()
        total = self.audio.duration_sec
        
        if total > 0 and current >= total:
            self.action_next_song()
            return

        if total > 0:
            pct = current / total
            self.query_one("#progress", ManualProgressBar).update_progress(pct)
            c_m, c_s = divmod(int(current), 60)
            t_m, t_s = divmod(int(total), 60)
            self.query_one("#time-label").update(f"{c_m:02}:{c_s:02} / {t_m:02}:{t_s:02}")

    def watch_current_metadata(self, meta):
        self.query_one("#meta-title").update(meta["title"])
        self.query_one("#meta-artist").update(meta["artist"])
        alb = meta["album"]
        if alb and alb != "-":
            self.query_one("#meta-album").update(alb)
            self.query_one("#meta-album").styles.display = "block"
        else:
            self.query_one("#meta-album").styles.display = "none"

    def on_seek_request(self, pct):
        if self.audio.duration_sec > 0 and self.audio.current_file:
            self.is_seeking = True
            self.query_one("#progress", ManualProgressBar).update_progress(pct)
            target_sec = self.audio.duration_sec * pct
            self.audio.load_and_play(self.audio.current_file, start_time=target_sec)
            self.is_playing = True
            self.query_one("#btn-play").label = "PAUSE"
            self.set_timer(0.2, lambda: setattr(self, 'is_seeking', False))

    @on(ListView.Selected)
    def play_track_handler(self, event):
        idx = self.query_one("#track-list").index
        self.play_index(idx)

    def play_index(self, idx):
        if 0 <= idx < len(self.playlist):
            self.current_idx = idx
            path = self.playlist[idx]
            self.audio.load_and_play(str(path))
            self.is_playing = True
            self.query_one("#btn-play").label = "PAUSE"
            self.current_metadata = {"title": path.stem, "artist": "Loading...", "album": ""}
            self.query_one("#album-art").update("") 
            self.process_heavy_tasks(path)

    def action_next_song(self):
        if self.playlist:
            idx = (self.current_idx + 1) % len(self.playlist)
            self.query_one("#track-list").index = idx
            self.play_index(idx)

    @work(thread=True)
    def process_heavy_tasks(self, path):
        self.audio.background_analyze(path)
        self.extract_metadata_and_cover(path)

    def extract_metadata_and_cover(self, path):
        try:
            f = MutagenFile(path)
            title = path.stem
            artist = "Unknown Artist"
            album = "-"
            
            # 1. 提取文本
            if f and f.tags:
                def get_text(k):
                    if k in f.tags:
                        val = f.tags[k]
                        if hasattr(val, 'text'): return str(val.text[0])
                        return str(val)
                    return None
                t = get_text('TIT2') or get_text('title')
                if t: title = t
                a = get_text('TPE1') or get_text('artist')
                if a: artist = a
                l = get_text('TALB') or get_text('album')
                if l: album = l
            
            self.current_metadata = {"title": title, "artist": artist, "album": album}

            # 2. 提取封面 (你的诊断结果表明 getall('APIC') 可行)
            artwork_data = None
            
            if str(path).lower().endswith(".mp3"):
                try:
                    audio_id3 = ID3(path)
                    apic_frames = audio_id3.getall("APIC")
                    if apic_frames:
                        artwork_data = apic_frames[0].data
                except Exception: pass

            if not artwork_data and str(path).lower().endswith(".flac"):
                if hasattr(f, 'pictures') and f.pictures:
                    artwork_data = f.pictures[0].data

            # 3. 渲染
            if artwork_data:
                try:
                    img = Image.open(io.BytesIO(artwork_data)).convert("RGB")
                    self.render_high_res_ascii(img)
                except Exception as e:
                    # 渲染失败则生成随机图
                    self.generate_procedural_art(title)
            else:
                self.generate_procedural_art(title)

        except Exception as e:
            self.current_metadata = {"title": path.stem, "artist": "Error", "album": ""}
            self.generate_procedural_art(path.stem)

    def generate_procedural_art(self, text):
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        rng = random.Random(seed)
        hex_c = f"#{rng.randint(50,255):02x}{rng.randint(50,255):02x}{rng.randint(50,255):02x}"
        w, h = 44, 20
        lines = []
        for _ in range(h):
            line = ""
            for _ in range(w):
                if rng.random() > 0.8: line += f"[{hex_c}]·[/]"
                else: line += " "
            lines.append(line)
        self.call_from_thread(self.update_cover_ui, "\n".join(lines), hex_c)

    # === [关键修复] 使用十六进制颜色代码 ===
    def render_high_res_ascii(self, img):
        try:
            w_char, h_char = 44, 20 
            img = img.resize((w_char, h_char * 2), Image.Resampling.LANCZOS)
            pixels = img.load()
            
            # 计算主色调
            center_px = img.getpixel((w_char//2, h_char))
            main_color = f"#{center_px[0]:02x}{center_px[1]:02x}{center_px[2]:02x}"

            textual_str = ""
            for y in range(0, h_char * 2, 2):
                for x in range(w_char):
                    top = pixels[x, y]
                    bot = pixels[x, y+1]
                    # 修复点：强制使用 #RRGGBB 格式
                    top_hex = f"#{top[0]:02x}{top[1]:02x}{top[2]:02x}"
                    bot_hex = f"#{bot[0]:02x}{bot[1]:02x}{bot[2]:02x}"
                    # 修复点：使用标准的 Rich 风格语法
                    textual_str += f"[{top_hex} on {bot_hex}]▀[/]"
                textual_str += "\n"

            self.call_from_thread(self.update_cover_ui, textual_str, main_color)
            
        except Exception as e:
            # 如果出错，显示红色的 ERROR
            self.call_from_thread(self.update_cover_ui, f"\n\n[red bold]RENDER ERROR:\n{str(e)[:20]}...[/]", "#ff0000")

    def update_cover_ui(self, art_str, theme_color):
        self.query_one("#album-art").update(art_str)
        self.query_one("#album-art").styles.border = ("heavy", theme_color)
        self.query_one("#spectrum").styles.color = theme_color
        self.query_one("#btn-play").styles.border = ("solid", theme_color)
        self.query_one("#meta-artist").styles.color = theme_color

    @on(Button.Pressed)
    def handle_buttons(self, event):
        bid = event.button.id
        if bid == "btn-play":
            self.action_toggle_play()
        elif bid == "btn-next":
            self.action_next_song()
        elif bid == "btn-prev":
            if self.playlist:
                idx = (self.current_idx - 1) % len(self.playlist)
                self.query_one("#track-list").index = idx
                self.play_index(idx)

    def action_toggle_play(self):
        if self.is_playing:
            pygame.mixer.music.pause()
            self.query_one("#btn-play").label = "PLAY"
            self.is_playing = False
        else:
            pygame.mixer.music.unpause()
            self.query_one("#btn-play").label = "PAUSE"
            self.is_playing = True

if __name__ == "__main__":
    app = NeonPlayer()
    app.run()
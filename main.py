import os
import io
import time
import random
import hashlib
from pathlib import Path

# UI
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, ListView, ListItem, Label, Button, Static, Sparkline
from textual import on, work
from textual.binding import Binding
from textual.events import Click

# Audio & Data
import pygame
import numpy as np
from pydub import AudioSegment
from PIL import Image, ImageDraw
from mutagen import File as MutagenFile
from mutagen.flac import FLAC

# --- 路径配置 ---
MUSIC_DIR = Path("J:\音乐\Hitorie") 
# ----------------

class AudioEngine:
    def __init__(self):
        pygame.mixer.init(frequency=44100)
        self.raw_data = None
        self.frame_rate = 44100
        self.duration_sec = 0
        self.analyzing = False
        self.start_offset = 0
        # [修复] 初始化变量
        self.current_file = None

    def load_and_play(self, path, start_time=0):
        self.analyzing = True
        self.start_offset = start_time
        # [修复] 记录当前文件，供进度条跳转使用
        self.current_file = path 
        self.raw_data = None 
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
            
            # 读取波形
            audio = AudioSegment.from_file(path)
            if len(audio) > 60000: audio = audio[:60000]
            self.frame_rate = audio.frame_rate
            self.raw_data = np.array(audio.set_channels(1).get_array_of_samples())
        except Exception as e: pass
        finally: self.analyzing = False

    def get_spectrum(self, bars=60):
        if self.raw_data is None:
            return [random.randint(1, 3) for _ in range(bars)] if self.analyzing else [0]*bars
        
        current_abs = self.get_current_time()
        display_time = current_abs % 60
        idx = int(display_time * self.frame_rate)
        chunk = 2048
        
        if idx + chunk > len(self.raw_data): return [0]*bars
        
        data = np.abs(self.raw_data[idx:idx+chunk])
        step = len(data) // bars
        res = [int(np.mean(data[i*step:(i+1)*step])) for i in range(bars)]
        return [min(x // 100, 8) for x in res]

# --- 手写进度条组件 ---
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
        
        # 手绘字符: 蓝色实线 + 白色滑块 + 灰色虚线
        bar = f"[{'#007acc'}]" + ("━" * filled_len) + "[/]" + \
              f"[{'#ffffff'}]●[/]" + \
              f"[{'#333333'}]" + ("─" * (width - filled_len - 1)) + "[/]"
        return bar

    def on_click(self, event: Click):
        pct = event.x / self.size.width
        self.app.on_seek_request(pct)

class NeonPlayer(App):
    CSS = """
    Screen { layout: horizontal; background: #111111; }
    
    #sidebar { 
        width: 30%; height: 100%; 
        background: #1a1a1a; 
        border-right: solid #333333; 
    }
    
    #main-view { width: 1fr; height: 100%; align: center middle; padding: 1; }

    #album-art { 
        height: 14; width: 46; 
        border: heavy #444444;
        margin-bottom: 1; text-align: center; color: #666666;
    }

    #track-title { text-style: bold; color: #ffffff; margin-top: 1; }
    #time-label { color: #888888; margin-bottom: 1; }

    ManualProgressBar { 
        width: 100%; 
        height: 1; 
        margin: 1 0;
    }
    
    Sparkline { height: 4; width: 100%; margin: 1 0; color: #007acc; }
    
    #controls { layout: horizontal; align: center middle; height: 3; margin-top: 1; }
    
    Button {
        min-width: 12;
        height: 3;
        margin: 0 1;
        border: none;
        background: #222222;
        color: #eeeeee;
        text-style: bold;
    }
    Button:hover { background: #007acc; color: #ffffff; }
    """

    BINDINGS = [
        Binding("space", "toggle_play", "播放/暂停"),
        Binding("left", "seek_backward", "-5s"),
        Binding("right", "seek_forward", "+5s"),
        Binding("q", "quit", "退出")
    ]

    def __init__(self):
        super().__init__()
        self.audio = AudioEngine()
        self.playlist = []
        self.current_idx = -1
        self.is_playing = False
        self.is_seeking = False

    def compose(self) -> ComposeResult:
        with Container(id="sidebar"):
            yield Label("  PLAYLIST", classes="title")
            yield ListView(id="track-list")
        
        with Vertical(id="main-view"):
            yield Label("\n\n\n[ No Data ]", id="album-art")
            
            with Vertical(classes="info-box"):
                yield Label("Waiting...", id="track-title")
                yield Label("00:00 / 00:00", id="time-label")
            
            yield ManualProgressBar(id="progress")
            yield Sparkline(summary_function=max, id="spectrum")
            
            with Container(id="controls"):
                yield Button("PREV", id="btn-prev")
                yield Button("PLAY", id="btn-play", variant="success")
                yield Button("NEXT", id="btn-next")
            
        yield Footer()

    def on_mount(self):
        self.load_files()
        self.set_interval(0.1, self.update_ui_tick)

    def load_files(self):
        lv = self.query_one("#track-list", ListView)
        exts = ["*.mp3", "*.flac", "*.wav"]
        if MUSIC_DIR.exists():
            files = []
            for ext in exts: files.extend(list(MUSIC_DIR.glob(ext)))
            files.sort(key=lambda x: x.name)
            for f in files:
                self.playlist.append(f)
                lv.append(ListItem(Label(f" {f.name}")))

    def update_ui_tick(self):
        if not self.is_playing or self.is_seeking: return
        
        data = self.audio.get_spectrum(bars=80)
        self.query_one("#spectrum", Sparkline).data = data
        
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

    def on_seek_request(self, pct):
        # [修复] 增加判断，确保当前有正在播放的文件，否则点击无效
        if self.audio.duration_sec > 0 and self.audio.current_file:
            self.is_seeking = True
            self.query_one("#progress", ManualProgressBar).update_progress(pct)
            
            target_sec = self.audio.duration_sec * pct
            self.audio.load_and_play(self.audio.current_file, start_time=target_sec)
            self.is_playing = True
            self.query_one("#btn-play").label = "PAUSE"
            self.set_timer(0.5, lambda: setattr(self, 'is_seeking', False))

    def action_seek_forward(self):
        if self.audio.duration_sec:
            t = self.audio.get_current_time() + 5
            self.on_seek_request(t / self.audio.duration_sec)
            
    def action_seek_backward(self):
        if self.audio.duration_sec:
            t = self.audio.get_current_time() - 5
            self.on_seek_request(t / self.audio.duration_sec)

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
            self.query_one("#track-title").update(path.stem)
            self.query_one("#btn-play").label = "PAUSE"
            self.process_heavy_tasks(path)

    @work(thread=True)
    def process_heavy_tasks(self, path):
        self.audio.background_analyze(path)
        self.extract_and_render_cover(path)

    def extract_and_render_cover(self, path):
        try:
            artwork = None
            f = MutagenFile(path)
            
            if f and f.tags:
                for key, value in f.tags.items():
                    k_str = str(key).upper()
                    if any(x in k_str for x in ['APIC', 'COVR', 'PICTURE', 'IMG']):
                        if hasattr(value, 'data'): artwork = value.data
                        elif isinstance(value, list) and hasattr(value[0], 'data'): artwork = value[0].data
                        if artwork: break

            if not artwork:
                self.generate_procedural_art(path.stem)
                return

            img = Image.open(io.BytesIO(artwork)).convert("RGB")
            self.render_ascii_art(img)

        except Exception as e:
            print(f"Cover Error: {e}")
            self.generate_procedural_art(path.stem)

    def generate_procedural_art(self, text):
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        rng = random.Random(seed)
        
        base_r = rng.randint(50, 200)
        base_g = rng.randint(50, 200)
        base_b = rng.randint(50, 200)
        hex_color = f"#{base_r:02x}{base_g:02x}{base_b:02x}"
        
        w, h = 46, 14
        ascii_str = ""
        for y in range(h):
            for x in range(w):
                if rng.random() > 0.5: ascii_str += f"[{hex_color}]█[/]"
                else: ascii_str += f"[{hex_color}] [/]"
            ascii_str += "\n"
        
        self.call_from_thread(self.update_cover_display, ascii_str, hex_color)

    def render_ascii_art(self, img):
        center = img.getpixel((img.width//2, img.height//2))
        hex_color = f"#{center[0]:02x}{center[1]:02x}{center[2]:02x}"
        
        w, h = 46, 14
        img = img.resize((w, h), Image.Resampling.NEAREST)
        pixels = img.load()
        ascii_str = ""
        for y in range(h):
            for x in range(w):
                r, g, b = pixels[x, y]
                ascii_str += f"[{r},{g},{b}]██[/]"
            ascii_str += "\n"
        
        self.call_from_thread(self.update_cover_display, ascii_str, hex_color)

    def update_cover_display(self, art_str, theme_color):
        self.query_one("#album-art").update(art_str)
        self.query_one("#spectrum").styles.color = theme_color
        self.query_one("#btn-play").styles.background = theme_color
        self.query_one("#sidebar").styles.border_right = ("solid", theme_color)
        self.query_one("#album-art").styles.border = ("heavy", theme_color)

    @on(Button.Pressed)
    def handle_buttons(self, event):
        bid = event.button.id
        if bid == "btn-play":
            self.action_toggle_play()
        elif bid == "btn-next":
            if self.playlist:
                idx = (self.current_idx + 1) % len(self.playlist)
                self.play_index(idx)
        elif bid == "btn-prev":
            if self.playlist:
                idx = (self.current_idx - 1) % len(self.playlist)
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
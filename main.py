import os
import io
import time
import random
import hashlib
import tempfile
import threading
import math
import argparse
from pathlib import Path

# UI
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, ListView, ListItem, Label, Button, Sparkline
from textual import on, work
from textual.binding import Binding
from textual.events import Click, Resize
from textual.reactive import reactive

# Audio & Data
import pygame
import numpy as np
from pydub import AudioSegment
from PIL import Image, ImageEnhance

# Networking
import requests 

# Metadata
from mutagen import File as MutagenFile
from mutagen.id3 import ID3 

# 配色
THEME_COLOR = "#00ff9d" 

# === 1. 音频引擎 (保持稳定版) ===
class AudioEngine:
    def __init__(self):
        pygame.mixer.init(frequency=44100)
        self.raw_data = None
        self.frame_rate = 11025 
        self.duration_sec = 0
        self.analyzing = False
        self.start_offset = 0
        self.fallback_mode = False
        self.chunk_size = 2048
        self.current_file = None 

    def load_and_play(self, path, start_time=0):
        self.start_offset = start_time
        self.current_file = path 
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play(start=start_time)
        except Exception as e:
            print(f"Play Error: {e}")

    def get_current_time(self):
        try:
            if pygame.mixer.music.get_busy():
                return self.start_offset + (pygame.mixer.music.get_pos() / 1000.0)
        except: pass
        return self.start_offset

    def background_analyze(self, path):
        self.analyzing = True
        self.raw_data = None
        self.fallback_mode = False
        try:
            f = MutagenFile(path)
            if f and hasattr(f, 'info'):
                self.duration_sec = f.info.length
            
            audio = AudioSegment.from_file(path)
            audio = audio.set_frame_rate(11025).set_channels(1)
            self.frame_rate = 11025
            self.raw_data = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)
            
        except Exception:
            self.fallback_mode = True
        finally:
            self.analyzing = False

    def get_spectrum(self, bars=60):
        if self.analyzing:
            return [random.randint(1, 2) for _ in range(bars)]
        if self.fallback_mode or self.raw_data is None:
            t = time.time()
            return [int(abs(math.sin(t * 3 + i * 0.2)) * 6) + 1 for i in range(bars)]

        current_time = self.get_current_time()
        idx = int(current_time * self.frame_rate)
        
        if idx >= len(self.raw_data): return [0] * bars
        
        end_idx = idx + self.chunk_size
        chunk = self.raw_data[idx:end_idx]
        
        if len(chunk) == 0: return [0] * bars

        chunk = np.abs(chunk)
        step = len(chunk) // bars
        if step < 1: step = 1
        
        res = []
        for i in range(bars):
            start = i * step
            segment = chunk[start : start + step]
            if len(segment) > 0:
                val = np.mean(segment)
                normalized = val / 32768.0
                height = (normalized ** 0.5) * 20 
                res.append(max(0, min(int(height), 8)))
            else:
                res.append(0)
        return res

# === UI 组件 ===
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
    Screen { background: #0f0f0f; }
    
    #sidebar { 
        dock: left;
        width: 30; height: 100%; 
        background: #141414; 
        border-right: solid #333333; 
    }
    
    .list-header { 
        padding: 1; text-align: center; text-style: bold; 
        background: #1a1a1a; color: #888888; 
    }
    
    ListView { background: #141414; }
    ListItem { padding: 1 2; color: #aaaaaa; }
    ListItem:hover { background: #222222; }
    ListItem.-selected { background: #1a1a1a; color: #00ff9d; border-left: solid #00ff9d; }

    #main-view { 
        width: 1fr; height: 100%; 
        layout: vertical;
        overflow: hidden; 
    }

    /* 动态容器 */
    #info-container {
        width: 100%; 
        height: 1fr; 
        align: center middle; 
        padding: 0 1; /* 左右留一点空隙 */
        overflow: hidden; /* 溢出隐藏是必要的，防止撑破布局 */
    }

    /* 封面图 */
    #album-art { 
        height: auto; 
        width: auto;
        border: heavy #333333; 
        margin-bottom: 1; /* 底部留出 1 行间距给文字 */
        text-align: center; 
        background: #000000; 
        color: #00ff9d;
        /* 关键：防止封面无限撑大 */
        box-sizing: border-box; 
    }

    /* 元数据区域：固定最小高度，防止被挤压成 0 */
    #meta-box {
        height: auto;
        min-height: 4; 
        align: center middle;
    }

    #meta-title { text-style: bold; color: #ffffff; text-align: center; }
    #meta-artist { color: #00ff9d; text-align: center; }
    #meta-album { color: #666666; text-align: center; text-style: italic; display: none; }

    #bottom-pane {
        dock: bottom; 
        height: auto; 
        min-height: 9; 
        background: #0f0f0f; 
        padding: 1 4;
        border-top: solid #222222;
    }

    #time-label { color: #555555; text-align: right; width: 100%; }
    ManualProgressBar { width: 100%; height: 1; margin: 0; }
    Sparkline { height: 3; width: 100%; margin: 1 0; color: #00ff9d; opacity: 60%; }
    
    #controls { 
        layout: horizontal; align: center middle; 
        height: 3; margin-top: 1; 
    }
    
    Button {
        min-width: 14; height: 3; margin: 0 1; border: none;
        background: #222222; color: #eeeeee;
        content-align: center middle;
    }
    Button:hover { background: #00ff9d; color: #000000; }
    #btn-play { background: #2a2a2a; border: solid #00ff9d; }
    """

    BINDINGS = [
        Binding("space", "toggle_play", "Play"),
        Binding("q", "quit", "Quit"),
        Binding("o", "open_cover", "Artwork") 
    ]

    current_metadata = reactive({"title": "Waiting...", "artist": "-", "album": "-"})
    
    # 核心数据
    current_pil_image = None
    current_cover_bytes = None

    def __init__(self, music_dir):
        super().__init__()
        self.music_dir = Path(music_dir).resolve()
        self.audio = AudioEngine()
        self.playlist = []
        self.current_idx = -1
        self.is_playing = False
        self.is_seeking = False
        self._resize_timer = None

    def compose(self) -> ComposeResult:
        with Container(id="sidebar"):
            folder_name = self.music_dir.name if self.music_dir.name else "ROOT"
            yield Label(f"DIR: {folder_name}", classes="list-header")
            yield ListView(id="track-list")
        
        with Container(id="main-view"):
            with Vertical(id="info-container"):
                yield Label("", id="album-art")
                with Vertical(id="meta-box"):
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
        yield Footer()

    def on_mount(self):
        self.load_files()
        self.set_interval(0.05, self.update_ui_tick)

    # === [优化] 响应式布局：主线程计算尺寸 -> 子线程生成图片 ===
    def on_resize(self, event: Resize):
        if self._resize_timer:
            self._resize_timer.stop()
        # 防抖：300ms 后执行重绘
        self._resize_timer = self.set_timer(0.3, self.trigger_render_pipeline)

    def trigger_render_pipeline(self):
        """主线程任务：计算可用空间，然后派发给 Worker"""
        if not self.current_pil_image:
            return

        # 1. 计算尺寸 (必须在主线程)
        screen_height = self.size.height
        
        # === [核心修复] ===
        # 屏幕总高 
        # - Header(1) - Footer(1) 
        # - BottomPane(9) (底部控制区)
        # - MetaBox(5) (预留给文字区域: 标题+歌手+专辑+Padding)
        # - Padding(2) (容器内边距)
        # - Border(2) (封面边框)
        # = 剩余给图片像素的高度
        # ------------------------------------------------
        # 算式：Screen - (1+1+9+5+2+2) = Screen - 20
        # 为了保险起见，我们减去 24，留出更多“呼吸空间”
        available_h = screen_height - 24
        
        # 限制高度范围：最小 4 行，最大 28 行
        # 如果 available_h <= 0，说明窗口太小了，强行给 4 行显示个大概
        target_h = max(4, min(available_h, 28))
        
        # 宽度比例 2.2 倍
        target_w = int(target_h * 2.2)

        # 2. 派发给后台线程
        self.run_worker(self.render_worker_task(self.current_pil_image, target_w, target_h), thread=True)

    async def render_worker_task(self, img, w, h):
        """Worker任务：执行缩放和字符生成"""
        try:
            # 缩放 (耗时操作)
            img_resized = img.resize((w, h * 2), Image.Resampling.BILINEAR)
            pixels = img_resized.load()
            
            center_px = img.getpixel((img.width//2, img.height//2))
            main_color = f"#{center_px[0]:02x}{center_px[1]:02x}{center_px[2]:02x}"

            lines = []
            for y in range(0, h * 2, 2):
                row_parts = []
                for x in range(w):
                    # 安全检查，防止越界
                    if x >= img_resized.width or y+1 >= img_resized.height:
                        continue
                    
                    top = pixels[x, y]
                    bot = pixels[x, y+1]
                    top_hex = f"#{top[0]:02x}{top[1]:02x}{top[2]:02x}"
                    bot_hex = f"#{bot[0]:02x}{bot[1]:02x}{bot[2]:02x}"
                    row_parts.append(f"[{top_hex} on {bot_hex}]▀[/]")
                lines.append("".join(row_parts))
            
            final_str = "\n".join(lines)
            
            # 回到主线程更新 UI
            self.call_from_thread(self.update_cover_ui, final_str, main_color)
            
        except Exception as e:
            # print(f"Render Error: {e}") # 调试用
            self.call_from_thread(self.update_cover_ui, f"[red]Render Error:\n{str(e)}[/]", "#ff0000")

    def load_files(self):
        lv = self.query_one("#track-list", ListView)
        if self.music_dir.exists() and self.music_dir.is_dir():
            files = [p for p in self.music_dir.glob("*") if p.suffix.lower() in ['.mp3', '.flac', '.wav', '.ogg']]
            files.sort(key=lambda x: x.name)
            if not files: self.notify(f"No music found in {self.music_dir}", severity="error")
            for f in files:
                self.playlist.append(f)
                lv.append(ListItem(Label(f" {f.name}")))
        else:
            self.notify(f"Invalid Directory: {self.music_dir}", severity="error")

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
            self.query_one("#album-art").update("\n\nLoading...") 
            self.process_heavy_tasks(path)

    def action_next_song(self):
        if self.playlist:
            idx = (self.current_idx + 1) % len(self.playlist)
            self.query_one("#track-list").index = idx
            self.play_index(idx)

    def action_open_cover(self):
        if self.current_cover_bytes:
            threading.Thread(target=self._open_image_file, args=(self.current_cover_bytes,)).start()
        else:
            self.notify("No cover image available", severity="warning")

    def _open_image_file(self, data):
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            os.startfile(tmp_path)
        except Exception as e:
            print(f"Open Error: {e}")

    @work(thread=True)
    def process_heavy_tasks(self, path):
        self.audio.background_analyze(path)
        self.extract_and_fetch_cover(path)

    def extract_and_fetch_cover(self, path):
        self.current_cover_bytes = None
        self.current_pil_image = None
        
        try:
            f = MutagenFile(path)
            title = path.stem
            artist = "Unknown"
            album = "-"
            
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

            artwork_data = None
            if str(path).lower().endswith(".mp3"):
                try:
                    audio_id3 = ID3(path)
                    apic_frames = audio_id3.getall("APIC")
                    if apic_frames: artwork_data = apic_frames[0].data
                except: pass
            if not artwork_data and str(path).lower().endswith(".flac"):
                if hasattr(f, 'pictures') and f.pictures: artwork_data = f.pictures[0].data

            if not artwork_data:
                self.call_from_thread(self.query_one("#album-art").update, "\n\nSearching Online...")
                artwork_data = self.fetch_online_cover(artist, title)

            if artwork_data:
                self.current_cover_bytes = artwork_data
                try:
                    img = Image.open(io.BytesIO(artwork_data)).convert("RGB")
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.2)
                    self.current_pil_image = img
                    # [关键修复] 调用主线程触发渲染管线
                    self.call_from_thread(self.trigger_render_pipeline)
                except:
                    self.generate_procedural_art(title)
            else:
                self.generate_procedural_art(title)

        except Exception as e:
            self.generate_procedural_art(path.stem)

    def fetch_online_cover(self, artist, title):
        if artist == "Unknown" or not title: return None
        try:
            term = f"{artist} {title}"
            url = "https://itunes.apple.com/search"
            params = {"term": term, "media": "music", "entity": "song", "limit": 1}
            resp = requests.get(url, params=params, timeout=3)
            data = resp.json()
            if data['resultCount'] > 0:
                thumb_url = data['results'][0]['artworkUrl100']
                hq_url = thumb_url.replace("100x100", "600x600")
                img_resp = requests.get(hq_url, timeout=3)
                if img_resp.status_code == 200:
                    return img_resp.content
        except: pass
        return None

    def generate_procedural_art(self, text):
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        rng = random.Random(seed)
        hex_c = f"#{rng.randint(50,255):02x}{rng.randint(50,255):02x}{rng.randint(50,255):02x}"
        w, h = 48, 20
        lines = []
        for _ in range(h):
            line = ""
            for _ in range(w):
                if rng.random() > 0.8: line += f"[{hex_c}]·[/]"
                else: line += " "
            lines.append(line)
        self.call_from_thread(self.update_cover_ui, "\n".join(lines), hex_c)

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
    parser = argparse.ArgumentParser(description="Neon TUI Music Player")
    parser.add_argument("path", nargs="?", default=".", help="Path to music")
    args = parser.parse_args()
    app = NeonPlayer(music_dir=args.path)
    app.run()
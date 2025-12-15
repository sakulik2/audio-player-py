import os
# 屏蔽 Pygame 欢迎信息
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

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
from PIL import Image, ImageEnhance, ImageDraw 

# Networking
import requests 

# Metadata
from mutagen import File as MutagenFile
from mutagen.id3 import ID3 
from mutagen.easyid3 import EasyID3 
from mutagen.flac import FLAC

# Color
THEME_COLOR = "#00ff9d" 

# === 1. AudioEngine ===
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
        time.sleep(0.02)
        self.raw_data = None
        self.fallback_mode = False
        try:
            f = MutagenFile(path)
            if f and hasattr(f, 'info'):
                self.duration_sec = f.info.length
            
            audio = AudioSegment.from_file(path)
            audio = audio.set_frame_rate(11025).set_channels(1)
            self.frame_rate = 11025
            
            new_data = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)
            self.raw_data = new_data
        except Exception:
            self.fallback_mode = True
        finally:
            self.analyzing = False

    def get_spectrum(self, bars=60):
        local_data = self.raw_data 
        if self.analyzing:
            return [random.randint(1, 2) for _ in range(bars)]
        if self.fallback_mode or local_data is None:
            t = time.time()
            return [int(abs(math.sin(t * 3 + i * 0.2)) * 6) + 1 for i in range(bars)]

        current_time = self.get_current_time()
        idx = int(current_time * self.frame_rate)
        if idx >= len(local_data): return [0] * bars
        end_idx = idx + self.chunk_size
        chunk = local_data[idx:end_idx]
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

# === UI ===
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
        overflow: hidden; 
    }

    #info-container {
        width: 100%; height: 1fr; 
        align: center middle; 
        padding: 0 1;
        position: relative; 
    }

    #album-art { 
        height: auto; 
        width: auto;
        border: heavy #333333; 
        text-align: center; background: #000000; color: #00ff9d;
        box-sizing: border-box;
        margin-bottom: 5; 
    }

    #meta-box {
        dock: bottom;
        height: auto;
        width: 100%;
        align: center middle;
        padding-bottom: 1;
        background: #0f0f0f 0%; 
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
        Binding("o", "open_cover", "Artwork"),
        Binding("m", "auto_tag", "AutoTag") 
    ]

    current_metadata = reactive({"title": "Waiting...", "artist": "-", "album": "-"})
    
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
            with Container(id="info-container"):
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

    def on_resize(self, event: Resize):
        if self._resize_timer: self._resize_timer.stop()
        self._resize_timer = self.set_timer(0.3, self.trigger_render_pipeline)

    def trigger_render_pipeline(self):
        if not self.current_pil_image: return
        screen_height = self.size.height

        available_h = screen_height - 23
        
        target_h = max(4, min(available_h, 40)) 
        target_w = int(target_h * 2.2)

        self.render_worker_task(self.current_pil_image, target_w, target_h)

    @work(thread=True)
    def render_worker_task(self, img, w, h):
        try:
            img_resized = img.resize((w, h * 2), Image.Resampling.BILINEAR)
            pixels = img_resized.load()
            
            center_px = img.getpixel((img.width//2, img.height//2))
            main_color = f"#{center_px[0]:02x}{center_px[1]:02x}{center_px[2]:02x}"
            
            lines = []
            for y in range(0, h * 2, 2):
                row_parts = []
                for x in range(w):
                    if x >= img_resized.width or y+1 >= img_resized.height: continue
                    top = pixels[x, y]
                    bot = pixels[x, y+1]
                    top_hex = f"#{top[0]:02x}{top[1]:02x}{top[2]:02x}"
                    bot_hex = f"#{bot[0]:02x}{bot[1]:02x}{bot[2]:02x}"
                    row_parts.append(f"[{top_hex} on {bot_hex}]▀[/]")
                lines.append("".join(row_parts))
            
            final_str = "\n".join(lines)
            self.call_from_thread(self.update_cover_ui, final_str, main_color)
        except Exception: pass

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
    
    # === Auto Tag ===
    def action_auto_tag(self):
        if not self.audio.current_file: return
        self.notify("Searching online metadata...", severity="information")
        self.perform_auto_tag(str(self.audio.current_file))

    @work(thread=True)
    def perform_auto_tag(self, path_str):
        try:
            path = Path(path_str)
            query = path.stem.replace("_", " ").replace("-", " ")
            url = "https://itunes.apple.com/search"
            params = {"term": query, "media": "music", "entity": "song", "limit": 1}
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            
            if data['resultCount'] > 0:
                track = data['results'][0]
                new_title = track.get('trackName', '')
                new_artist = track.get('artistName', '')
                new_album = track.get('collectionName', '')
                thumb_url = track.get('artworkUrl100', '')
                
                self.write_tags_to_file(path, new_title, new_artist, new_album)
                
                self.call_from_thread(self.update_metadata_ui, new_title, new_artist, new_album)
                
                if thumb_url:
                    hq_url = thumb_url.replace("100x100", "600x600")
                    img_resp = requests.get(hq_url, timeout=5)
                    if img_resp.status_code == 200:
                        img_data = img_resp.content
                        self.current_cover_bytes = img_data
                        
                        img = Image.open(io.BytesIO(img_data)).convert("RGB")
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(1.2)
                        
                        self.current_pil_image = img
                        self.call_from_thread(self.trigger_render_pipeline)

                self.notify(f"Tagged: {new_title}", severity="success")
            else:
                self.notify("No match found online.", severity="warning")
                
        except Exception as e:
            self.notify(f"Tag Error: {str(e)}", severity="error")

    def update_metadata_ui(self, title, artist, album):
        self.current_metadata = {"title": title, "artist": artist, "album": album}

    def write_tags_to_file(self, path, title, artist, album):
        ext = path.suffix.lower()
        try:
            if ext == ".mp3":
                try: tags = EasyID3(path)
                except: tags = EasyID3()
                tags['title'] = title
                tags['artist'] = artist
                tags['album'] = album
                tags.save(path)
            elif ext == ".flac":
                audio = FLAC(path)
                audio['title'] = title
                audio['artist'] = artist
                audio['album'] = album
                audio.save()
        except Exception: pass

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
                    self.call_from_thread(self.trigger_render_pipeline)
                except:
                    self.create_procedural_image(title)
            else:
                self.create_procedural_image(title)

        except Exception as e:
            self.create_procedural_image(path.stem)

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

    def create_procedural_image(self, text):
        """生成随机像素画作为 PIL Image，走统一的渲染管线，解决吞字和缩放问题"""
        try:
            seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
            rng = random.Random(seed)
            
            w, h = 24, 24 
            img = Image.new('RGB', (w, h), color='#000000')
            pixels = img.load()
            
            base_r = rng.randint(20, 200)
            base_g = rng.randint(20, 200)
            base_b = rng.randint(20, 200)
            
            for y in range(h):
                for x in range(w):
                    if rng.random() > 0.8: 
                        r = min(255, base_r + rng.randint(-30, 30))
                        g = min(255, base_g + rng.randint(-30, 30))
                        b = min(255, base_b + rng.randint(-30, 30))
                        pixels[x, y] = (r, g, b)
            
            self.current_pil_image = img
            self.call_from_thread(self.trigger_render_pipeline)
            
        except Exception: pass

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
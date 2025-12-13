## A music player

A cyberpunk-styled, responsive terminal music player written in Python. It features a real-time spectrum visualizer, high-resolution ASCII cover art rendering, and automatic metadata fetching.

![Screenshot](https://raw.githubusercontent.com/sakulik2/photo/main/WindowsTerminal_HnfmXwrlOF.png)

## ✨ Features

*   **Cyberpunk UI:** Built with [Textual](https://textual.textualize.io/), featuring a neon color scheme and smooth animations.
*   **Real-time Visualizer:** Fluid audio spectrum analyzer (frequency bars) that reacts to the music.
*   **Smart Cover Art:**
    *   Extracts embedded covers from ID3 (MP3) or FLAC tags.
    *   Auto-fetches high-res covers from the web if local tags are missing.
    *   Falls back to procedural hash-based pixel art if no cover is found.
    *   **High-Res ASCII:** Renders covers using half-block characters for double vertical resolution.
*   **Responsive Layout:** The UI adapts intelligently to window resizing, scaling the cover art while keeping metadata visible.
*   **Format Support:** Supports MP3, FLAC, WAV, and OGG.
*   **Mouse Support:** Click the progress bar to seek; click buttons to control playback.

## 🛠️ Installation

### 1. Prerequisites
You need **Python 3.8+** installed.

### 2. Install Python Libraries
Run the following command to install the required dependencies:

```bash
pip install textual pygame numpy pydub mutagen pillow requests
```

### 3. Install FFmpeg (Crucial!)
`pydub` requires FFmpeg to handle audio formats like MP3 and FLAC.
*   **Windows:** [Download FFmpeg](https://ffmpeg.org/download.html), extract it, and **add the `bin` folder to your System PATH**.
*   **Mac:** `brew install ffmpeg`
*   **Linux:** `sudo apt install ffmpeg`

## 🚀 Usage

### Basic Start
Open your terminal in the project folder and run:

```bash
python main.py
```
*This will scan the current directory for music files.*

### Open Specific Folder
You can specify a music directory path as an argument:

```bash
python main.py "C:\Users\Music\Favorite Songs"
```
*(Note: Use quotes if your path contains spaces).*

## 🎮 Controls

| Key / Action | Function |
| :--- | :--- |
| **Space** | Play / Pause |
| **Q** | Quit |
| **O** | **Open Artwork** (Opens the high-res cover in system viewer) |
| **Mouse Click** | Click the progress bar to seek |
| **Mouse Click** | Click PREV / PLAY / NEXT buttons |

## 💡 Tips

*   **Terminal Choice:** For the best visual experience (TrueColor support and correct font rendering), use **Windows Terminal**, **VS Code Terminal**, or **iTerm2**. Legacy `cmd.exe` or PowerShell ISE may not render colors correctly.
*   **Performance:** The player uses a separate thread for image processing to ensure the UI remains smooth during resizing or track switching.
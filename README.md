<h1 align="center"> tinyjam </h1>
<p align="center">
    <a href="#readme">
        <img alt="alicia queen keys" src="https://raw.githubusercontent.com/aryakaul/tinyjam/refs/heads/main/assets/queen.png">
    </a>
</p>
<p align="center"> 💃🏽- jam 2 tiny desks </p>

---

## Quick Start

Pick an artist list (one name per line). A starter list lives at
[`arya-curated`](https://raw.githubusercontent.com/aryakaul/tinyjam/main/arya-curated).

### macOS (Homebrew)
```bash
brew tap aryakaul/formulae
brew install tinyjam
wget https://raw.githubusercontent.com/aryakaul/tinyjam/main/arya-curated
tinyjam -l ./arya-curated -n  # stream without downloading
```

### Any platform (PyPI)
```bash
pip install tinyjam
# tinyjam expects `mpv` and `yt-dlp` on your PATH
wget https://raw.githubusercontent.com/aryakaul/tinyjam/main/arya-curated
tinyjam -l ./arya-curated
```

### From source
```bash
git clone https://github.com/aryakaul/tinyjam.git
cd tinyjam
pip install --upgrade pip build
pip install -e .
tinyjam -l ./arya-curated -n
```

---

## Command Line

```
tinyjam --help

    -l, --list        File with one artist per line (required)
    -o, --output      Download folder (default: ./jamsesh)
    -n, --nodownload  Stream directly via mpv/yt-dlp
    -f, --force       Re-download even if files exist
    -j, --jobs        Parallel downloads (0 = auto)
    -S, --subtitles   Preferred subtitle language / regex (default: en)
    -c, --color       Watch in color (default: grayscale)
    --dry-run         Print commands without executing
    -v, --verbose     Extra logging
```

Tinyjam keeps a download cache, retries through `yt-dlp`, and can fetch manual subtitles when a Tiny Desk isn’t in your preferred language. Use `--nodownload` to shuffle a curated list straight from YouTube, or let it fill `./jamsesh` and loop locally via mpv.

Enjoy the desks ✨

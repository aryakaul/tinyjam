<h1 align="center"> tinyjam </h1>
<p align="center">
    <a href="#readme">
        <img alt="alicia queen keys" src="https://raw.githubusercontent.com/aryakaul/tinyjam/refs/heads/main/assets/queen.png">
    </a>
</p>
<p align="center"> 💃🏽- jam 2 tiny desks </p>

---

## Quick Start

This is a CLI tool to scrape and listen to a curated collection of Tiny Desks. To use it,
first build a artist list (one name per line). For example,

```
> echo "Anderson .Paak" >> jamlist
> tinyjam -l ./jamlist -o ~/videos/tinydesk
```

My list lives in `/assets/aryapproved/masterlist`. Launching `tinyjam` with no options
will automagically grab it and begin streaming it (functionally the same as
`tinyjam -l ./masterlist -n`)

You can also pull out performance segments by specifying timestamps i.e.
```
> echo "Nick Hakim (00:00-05:22)" >> mixtape
> tinyjam -l ./mixtape -o ~/videos/tinydesk
```
We also have support for subtitle downloading, playlist creation, and cacheing! JAMON

---

## Installation

### macOS (Homebrew)
```bash
brew tap aryakaul/formulae
brew install tinyjam
tinyjam 
```

### Any platform (PyPI)
```bash
pip install tinyjam
# tinyjam expects `mpv` and `yt-dlp` on your PATH
# ffmpeg is required when using timestamp features
tinyjam
```

### From source
```bash
git clone https://github.com/aryakaul/tinyjam.git
cd tinyjam
pip install --upgrade pip build
pip install -e .
tinyjam -l ./arya-curated -o ~/videos/tinydesk
```

---

## Command Line

```
usage: tinyjam [-h] -l LIST [-o OUTPUT] [-n] [-p {shuffle,forward,reverse}] [-f] [-c] [--dry-run] [-v]
               [-j JOBS] [-S SUBTITLES]

Jam to tiny desks with tinyjam (Python edition)

options:
  -h, --help            show this help message and exit
  -l LIST, --list LIST  path to artist list file (one per line) (default: None)
  -o OUTPUT, --output OUTPUT
                        directory to place downloads (default: jamsesh)
  -n, --nodownload      stream directly from YouTube instead of downloading (default: False)
  -p {shuffle,forward,reverse}, --playlist-order {shuffle,forward,reverse}
                        playback order: shuffle (default), jam list order (forward), or reverse jam list
                        order (default: shuffle)
  -f, --force           clear downloads and archive before downloading (default: False)
  -c, --color           display videos in color (default is grayscale) (default: False)
  --dry-run, --noop     print commands without invoking mpv or yt-dlp (default: False)
  -v, --verbose         increase logging verbosity (default: False)
  -j JOBS, --jobs JOBS  number of parallel downloads to run (0 = auto / use all available cores)
                        (default: 0)
  -S SUBTITLES, --subtitles SUBTITLES
                        preferred subtitle language (downloads only when the Tiny Desk is in a different
                        language) (default: en)
```

Tinyjam keeps a download cache, retries through `yt-dlp`, and can fetch manual subtitles when a Tiny Desk isn't in your preferred language. Use `--nodownload` to shuffle a curated list straight from YouTube, or let it fill `./jamsesh` and loop locally via mpv. Running `tinyjam` with zero flags defaults to streaming my curated list.

Enjoy the desks ✨

---

## Versioning

Releases are logged in `CHANGELOG.md` at the repo root. See the changelog for version history and release notes.

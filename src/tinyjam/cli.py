#!/usr/bin/env python3
"""
Python port of the tinyjam Bash script.

Provides the same behaviour as the shell implementation while adding structured
logging and a verbosity flag backed by Loguru.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from loguru import logger
from tqdm import tqdm

YT_CHANNEL_ID = "UC4eYXhJI4-7wSWc8UNRwD4A"
STANDARD_OPS: Sequence[str] = (
    "--contrast=-15",
    "--geometry=80%",
    "--video-zoom=-0.25",
)
DEFAULT_OUTPUT = Path("./jamsesh")
ARCHIVE_NAME = ".tinyjam_archive.txt"
FAILURES_NAME = ".tinyjam_failures"
ARTIST_CACHE_NAME = ".tinyjam_artists.txt"
INVALID_FILENAME_CHARS = re.compile(r"[\\/:*?\"<>|]+")
TITLE_SUFFIX_RE = re.compile(r"([:\-\|]\s*)?\(?tiny desk.*$", re.IGNORECASE)
YEAR_RE = re.compile(r"(\d{4})")
ENGLISH_PREFIXES = ("en", "eng")
SUB_LINE_RE = re.compile(r"^([A-Za-z0-9][\w\.-]*)\s")


def configure_logging(verbose: bool) -> None:
    """Configure loguru logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<level>{level: <8}</level> | <cyan>{message}</cyan>",
    )


def require_cmd(cmd: str) -> None:
    """Ensure a required command exists in PATH."""
    if shutil.which(cmd) is None:
        logger.error("required command '{}' not found in PATH", cmd)
        sys.exit(1)


def shlex_join(parts: Sequence[str]) -> str:
    """Join a command for display, preserving shell quoting."""
    try:
        return shlex.join(parts)
    except AttributeError:
        # Python < 3.8 fallback (not expected, but kept for completeness).
        return " ".join(shlex.quote(part) for part in parts)


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Return items with duplicates removed while preserving order."""
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def sanitize_label(label: str) -> str:
    """Best-effort cleanup for jamlist labels."""
    cleaned = INVALID_FILENAME_CHARS.sub("_", label.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "tinyjam"


def sanitize_title(raw_title: str) -> str:
    """Derive a filename-safe title from the YouTube metadata."""
    base = raw_title.strip()
    cleaned = TITLE_SUFFIX_RE.sub("", base).strip(" -:|")
    if not cleaned:
        cleaned = base
    cleaned = INVALID_FILENAME_CHARS.sub("_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "tinydesk"


def extract_year(entry: Dict[str, Any]) -> str:
    """Extract a 4-digit year from yt-dlp metadata."""
    for key in ("upload_date", "release_date", "timestamp"):
        value = entry.get(key)
        if isinstance(value, int):
            text = str(value)
        elif isinstance(value, str):
            text = value
        else:
            continue
        match = YEAR_RE.search(text)
        if match:
            return match.group(1)
    for key in ("release_year", "season_number", "year"):
        value = entry.get(key)
        if isinstance(value, int) and 1900 <= value <= 2100:
            return str(value)
    return "unknown"


def extract_language(entry: Dict[str, Any]) -> Optional[str]:
    language = entry.get("language")
    if isinstance(language, str) and language:
        return language.lower()
    return None


def _normalize_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    cleaned = lang.lower()
    if "-" in cleaned:
        cleaned = cleaned.split("-", 1)[0]
    return cleaned or None


def should_download_subs(video_language: Optional[str], desired_language: str) -> bool:
    normalized_video = _normalize_lang(video_language) or "en"
    normalized_target = _normalize_lang(desired_language) or "en"
    return normalized_video != normalized_target


def build_subtitle_args(target_language: str) -> List[str]:
    normalized_target = target_language or "en"
    return [
        "--write-subs",
        "--no-write-auto-subs",
        "--sub-langs",
        normalized_target,
        "--sub-format",
        "vtt",
    ]


@dataclass
class TinyJamContext:
    jamlist: Path
    output: Path
    nodownload: bool
    color: bool
    force: bool
    dry_run: bool
    verbose: bool
    jobs: int
    subtitle_lang: str
    playlist: List[str] = field(default_factory=list)
    pending_downloads: List[str] = field(default_factory=list)
    download_archive: Optional[Path] = None
    failure_log: Optional[Path] = None
    artist_cache: Optional[Path] = None
    download_errors: int = 0
    cached_queries: Dict[str, str] = field(default_factory=dict)
    cached_video_ids: Set[str] = field(default_factory=set)
    _failure_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _cache_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )


@dataclass(frozen=True)
class VideoSelection:
    video_id: str
    title: str
    url: str
    year: str
    language: Optional[str]


class TinyJam:
    def __init__(self, ctx: TinyJamContext) -> None:
        self.ctx = ctx

    def _effective_jobs(self) -> int:
        if self.ctx.jobs > 0:
            return self.ctx.jobs
        env_value = os.environ.get("PARALLEL_JOBS")
        jobs = 0
        if env_value:
            try:
                jobs = int(env_value)
            except ValueError:
                jobs = 0
        if jobs <= 0:
            jobs = os.cpu_count() or 1
        return max(1, jobs)

    def load_playlist(self) -> None:
        if not self.ctx.jamlist.is_file():
            logger.error("jam list '{}' not found", self.ctx.jamlist)
            sys.exit(1)

        items: List[str] = []
        with self.ctx.jamlist.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                items.append(line)

        if not items:
            logger.error("jam list '{}' is empty", self.ctx.jamlist)
            sys.exit(1)

        self.ctx.playlist = items

    def prepare_output(self) -> None:
        archive_path = self.ctx.output / ARCHIVE_NAME
        artist_cache = self.ctx.output / ARTIST_CACHE_NAME
        self.ctx.download_archive = archive_path
        self.ctx.artist_cache = artist_cache

        if self.ctx.dry_run:
            logger.info("dry run | skipping filesystem changes in {}", self.ctx.output)
            self.ctx.failure_log = None
            return

        self.ctx.output.mkdir(parents=True, exist_ok=True)
        failure_log = self.ctx.output / FAILURES_NAME
        with failure_log.open("w", encoding="utf-8"):
            pass
        self.ctx.failure_log = failure_log

        if self.ctx.force:
            logger.info(
                "force mode enabled | clearing downloads in {}",
                self.ctx.output,
            )
            if archive_path.exists():
                archive_path.unlink()
            if artist_cache.exists():
                artist_cache.unlink()
            for item in self.ctx.output.iterdir():
                if item.is_file():
                    try:
                        item.unlink()
                    except OSError as exc:
                        logger.warning("could not remove {}: {}", item, exc)

        archive_path.touch(exist_ok=True)
        artist_cache.touch(exist_ok=True)

    def compute_pending_downloads(self) -> None:
        unique_artists = dedupe_preserve_order(self.ctx.playlist)
        self.ctx.pending_downloads = unique_artists.copy()

        cache_path = self.ctx.artist_cache
        self.ctx.cached_queries = {}
        self.ctx.cached_video_ids = set()
        if cache_path is None or not cache_path.exists():
            return

        try:
            lines = cache_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            video_id = ""
            query = line
            if "\t" in line:
                video_id, query = line.split("\t", 1)
                video_id = video_id.strip()
                query = query.strip()
            if query:
                self.ctx.cached_queries[query] = video_id
            if video_id:
                self.ctx.cached_video_ids.add(video_id)

        self.ctx.pending_downloads = [
            artist for artist in unique_artists if artist not in self.ctx.cached_queries
        ]

    def record_failure(self, artist: str) -> None:
        if self.ctx.failure_log is None:
            return
        with (
            self.ctx._failure_lock,
            self.ctx.failure_log.open("a", encoding="utf-8") as handle,
        ):
            handle.write(f"{artist}\n")

    def mark_downloaded(self, artist: str, video_id: str) -> None:
        if self.ctx.dry_run or self.ctx.artist_cache is None:
            return
        with self.ctx._cache_lock:
            cache_path = self.ctx.artist_cache
            if cache_path is None:
                return
            existing_id = self.ctx.cached_queries.get(artist)
            if existing_id == video_id:
                return
            line = f"{video_id}\t{artist}\n" if video_id else f"{artist}\n"
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            self.ctx.cached_queries[artist] = video_id
            if video_id:
                self.ctx.cached_video_ids.add(video_id)

    def list_available_subtitles(self, url: str) -> List[str]:
        cmd = ["yt-dlp", "--skip-download", "--dump-single-json", url]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.debug(
                "subtitle metadata fetch failed | url: {} | stderr: {}",
                url,
                (result.stderr or "").strip(),
            )
            return []

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            logger.debug(
                "subtitle metadata invalid json | url: {} | error: {}",
                url,
                exc,
            )
            return []

        subtitles = payload.get("subtitles") or {}
        if not isinstance(subtitles, dict):
            subtitles = {}
        languages = sorted(subtitles.keys())
        logger.debug("manual subs detected | {} | {}", url, languages)
        return languages

    def select_subtitle_language(self, selection: VideoSelection) -> Optional[str]:
        available = self.list_available_subtitles(selection.url)
        if not available:
            return None
        pattern_text = self.ctx.subtitle_lang or "en"
        try:
            pattern = re.compile(pattern_text, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(pattern_text), re.IGNORECASE)
        for lang in available:
            if pattern.search(lang):
                return lang
        return None

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in re.split(r"\s+", text.lower()) if token]

    def _entry_on_channel(self, entry: Dict[str, Any]) -> bool:
        return entry.get("channel_id") == YT_CHANNEL_ID

    def _entry_has_tiny_desk(self, entry: Dict[str, Any]) -> bool:
        return "tiny desk" in (entry.get("title") or "").lower()

    def _entry_matches_artist_tokens(
        self, entry: Dict[str, Any], artist_tokens: List[str]
    ) -> bool:
        if not artist_tokens:
            return False
        title = (entry.get("title") or "").lower()
        return all(token in title for token in artist_tokens)

    def _choose_entry(
        self, entries: Iterable[Dict[str, Any]], artist: str
    ) -> Optional[Dict[str, Any]]:
        sanitized = [entry for entry in entries if isinstance(entry, dict)]
        if not sanitized:
            return None

        artist_tokens = self._tokenize(artist)
        best_entry: Optional[Dict[str, Any]] = None
        best_score: Tuple[int, int, int] = (-1, -1, -1)

        for entry in sanitized:
            score = (
                1 if self._entry_on_channel(entry) else 0,
                1 if self._entry_has_tiny_desk(entry) else 0,
                1 if self._entry_matches_artist_tokens(entry, artist_tokens) else 0,
            )
            if score > best_score:
                best_entry = entry
                best_score = score

        return best_entry

    def select_video(self, artist: str) -> Optional[VideoSelection]:
        query = artist.strip()
        search_expr = f"ytsearch15:npr tiny desk {query}"
        cmd = [
            "yt-dlp",
            "--dump-single-json",
            "--skip-download",
            "--no-progress",
            search_expr,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.warning(
                "search failed | {} | {}",
                artist,
                (result.stderr or result.stdout or "").strip(),
            )
            return None

        raw_output = result.stdout.strip()
        if not raw_output:
            logger.warning("empty search response | {}", artist)
            return None

        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            logger.warning("invalid search payload | {} | {}", artist, exc)
            logger.debug("search output for {}:\n{}", artist, raw_output)
            return None

        entries = payload.get("entries")
        if isinstance(entries, list) and entries:
            entry = self._choose_entry(entries, artist)
        elif payload.get("id"):
            entry = payload
        else:
            entry = None

        if not entry:
            logger.warning("no Tiny Desk match found | {}", artist)
            return None

        video_id = entry.get("id") or ""
        url = (
            entry.get("original_url")
            or entry.get("webpage_url")
            or entry.get("url")
            or ""
        )
        title = entry.get("title") or video_id
        if video_id and not url:
            url = f"https://www.youtube.com/watch?v={video_id}"

        if not video_id or not url:
            logger.warning("search result missing id or url | {} | {}", artist, entry)
            return None

        # Always build a canonical YouTube watch URL so later downloads never
        # re-trigger a search input like "ytsearch5:...".
        return VideoSelection(
            video_id=video_id,
            title=title,
            url=f"https://www.youtube.com/watch?v={video_id}",
            year=extract_year(entry),
            language=extract_language(entry),
        )

    def download_single(self, artist: str) -> bool:
        archive = str(self.ctx.download_archive or (self.ctx.output / ARCHIVE_NAME))

        if self.ctx.dry_run:
            search_expr = f"ytsearch15:npr tiny desk {artist.strip()}"
            logger.info(
                "dry run | {}",
                shlex_join(
                    [
                        "yt-dlp",
                        "--dump-single-json",
                        "--skip-download",
                        "--no-progress",
                        search_expr,
                    ]
                ),
            )
            logger.info(
                "dry run | {}",
                shlex_join(
                    [
                        "yt-dlp",
                        "<resolved-video-url>",
                        "--download-archive",
                        archive,
                        "--output",
                        str(
                            self.ctx.output / "<youtube-title>_[<id>]_(<year>).%(ext)s"
                        ),
                        "--restrict-filenames",
                        "--no-overwrites",
                        "--no-progress",
                        "-f",
                        "bestvideo[ext=webm]+bestaudio[ext=webm]/"
                        "bestvideo+bestaudio/best",
                    ]
                ),
            )
            return True

        selection = self.select_video(artist)
        if not selection:
            self.record_failure(artist)
            return False

        logger.debug(
            "language detection | {} -> video: {} | preferred: {}",
            artist,
            selection.language or "unknown",
            self.ctx.subtitle_lang,
        )

        title_label = sanitize_title(selection.title)
        year_label = selection.year if selection.year else "unknown"
        output_template = str(
            self.ctx.output
            / f"{title_label}_[{selection.video_id}]_({year_label}).%(ext)s"
        )
        subtitle_args: List[str] = []
        if should_download_subs(selection.language, self.ctx.subtitle_lang):
            matched_lang = self.select_subtitle_language(selection)
            if matched_lang:
                logger.debug(
                    "subtitle download requested | {} | lang: {}",
                    artist,
                    matched_lang,
                )
                subtitle_args = build_subtitle_args(matched_lang)
            else:
                logger.debug("subtitle download skipped (no match) | {}", artist)
        else:
            logger.debug("subtitle download skipped | {}", artist)
        if selection.video_id in self.ctx.cached_video_ids:
            self.mark_downloaded(artist, selection.video_id)
            return True

        cmd = [
            "yt-dlp",
            selection.url,
            "--match-filter",
            f'channel_id = {YT_CHANNEL_ID} & title ~= "(?i)tiny desk"',
            "--download-archive",
            archive,
            "--output",
            output_template,
            "--restrict-filenames",
            "--no-overwrites",
            "--no-progress",
            "-f",
            "bestvideo[ext=webm]+bestaudio[ext=webm]/bestvideo+bestaudio/best",
        ]
        cmd.extend(subtitle_args)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        output = result.stdout or ""
        if result.returncode != 0:
            logger.error("failed to download | {}", artist)
            if output:
                logger.debug("yt-dlp output for {}:\n{}", artist, output)
            self.record_failure(artist)
            return False

        self.mark_downloaded(artist, selection.video_id)
        if output:
            logger.debug("yt-dlp output for {}:\n{}", artist, output)
        return True

    def run_downloads(self) -> None:
        pending = self.ctx.pending_downloads
        if not pending:
            return

        jobs = self._effective_jobs()
        desc = "downloading"

        if self.ctx.dry_run or jobs < 2:
            with tqdm(total=len(pending), desc=desc, unit="jam") as progress:
                for artist in pending:
                    if not self.download_single(artist):
                        self.ctx.download_errors += 1
                    progress.update(1)
            return

        logger.info("using thread pool for downloads | jobs: {}", jobs)
        errors = 0
        with tqdm(total=len(pending), desc=desc, unit="jam") as progress:
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(self.download_single, artist): artist
                    for artist in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    artist = futures[future]
                    try:
                        success = future.result()
                    except Exception as exc:
                        logger.error("download raised exception | {} | {}", artist, exc)
                        success = False
                    if not success:
                        errors += 1
                    progress.update(1)

        self.ctx.download_errors += errors

    def play_stream(self) -> None:
        if not self.ctx.playlist:
            logger.info("playlist empty | nothing to play")
            return

        playlist = self.ctx.playlist[:]
        random.shuffle(playlist)

        color_flag = ["--saturation=20"] if self.ctx.color else ["--saturation=-100"]
        sub_flags = [
            "--sub-auto=fuzzy",
            f"--slang={self.ctx.subtitle_lang}",
        ]
        base_cmd = ["mpv", *sub_flags, *color_flag, *STANDARD_OPS]

        for artist in playlist:
            query = f"ytdl://ytsearch1:npr tiny desk {artist}"
            cmd = base_cmd + [query]
            logger.info("streaming | {}", artist)
            if self.ctx.dry_run:
                logger.info("dry run | {}", shlex_join(cmd))
                continue
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                logger.warning("mpv exited with error | {}", artist)

    def play_downloads(self) -> None:
        if not self.ctx.output.is_dir():
            logger.info("playback skipped | no output directory at {}", self.ctx.output)
            return

        valid_exts = {".mp4", ".webm", ".mkv", ".mov", ".m4v", ".mpg", ".mpeg"}
        files = sorted(
            [
                str(path)
                for path in self.ctx.output.iterdir()
                if path.is_file() and path.suffix.lower() in valid_exts
            ],
            key=lambda name: name.lower(),
        )
        if not files:
            logger.info("playback skipped | no files found in {}", self.ctx.output)
            return

        color_flag = ["--saturation=20"] if self.ctx.color else ["--saturation=-100"]
        sub_flags = [
            "--sub-auto=fuzzy",
            f"--slang={self.ctx.subtitle_lang}",
        ]
        cmd = [
            "mpv",
            *sub_flags,
            *color_flag,
            *STANDARD_OPS,
            "--loop-playlist",
            "--shuffle",
            *files,
        ]

        if self.ctx.dry_run:
            logger.info("dry run | {}", shlex_join(cmd))
            return

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.warning("mpv exited with error | {}", self.ctx.output)

    def summarize_failures(self) -> None:
        if self.ctx.download_errors <= 0:
            return

        if self.ctx.failure_log and self.ctx.failure_log.exists():
            logger.warning(
                "download issues | {} failure(s) recorded in {}",
                self.ctx.download_errors,
                self.ctx.failure_log,
            )
        else:
            logger.warning(
                "download issues | {} failure(s) encountered",
                self.ctx.download_errors,
            )

    def run(self) -> int:
        self.load_playlist()
        logger.info("output directory | {}", self.ctx.output)

        if self.ctx.nodownload:
            if not self.ctx.dry_run:
                require_cmd("mpv")
            self.play_stream()
            return 0

        if not self.ctx.dry_run:
            require_cmd("yt-dlp")
            require_cmd("mpv")
        else:
            logger.info("dry run | skipping dependency checks")

        self.prepare_output()
        self.compute_pending_downloads()
        if not self.ctx.pending_downloads:
            logger.info(
                "downloads up-to-date | all jams already cached in {}",
                self.ctx.output,
            )
        else:
            self.run_downloads()

        self.summarize_failures()
        self.play_downloads()

        if self.ctx.download_errors > 0:
            return 1
        return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> TinyJamContext:
    parser = argparse.ArgumentParser(
        description="Jam to tiny desks with tinyjam (Python edition)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l",
        "--list",
        required=True,
        type=Path,
        help="path to artist list file (one per line)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory to place downloads",
    )
    parser.add_argument(
        "-n",
        "--nodownload",
        action="store_true",
        help="stream directly from YouTube instead of downloading",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="clear downloads and archive before downloading",
    )
    parser.add_argument(
        "-c",
        "--color",
        action="store_true",
        help="display videos in color (default is grayscale)",
    )
    parser.add_argument(
        "--dry-run",
        "--noop",
        dest="dry_run",
        action="store_true",
        help="print commands without invoking mpv or yt-dlp",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="increase logging verbosity",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help=(
            "number of parallel downloads to run (0 = auto / use all available cores)"
        ),
    )
    parser.add_argument(
        "-S",
        "--subtitles",
        type=str,
        default="en",
        help=(
            "preferred subtitle language (downloads only when the Tiny Desk "
            "is in a different language)"
        ),
    )
    args = parser.parse_args(argv)

    ctx = TinyJamContext(
        jamlist=args.list,
        output=args.output,
        nodownload=args.nodownload,
        color=args.color,
        force=args.force,
        dry_run=args.dry_run,
        verbose=args.verbose,
        jobs=args.jobs,
        subtitle_lang=args.subtitles,
    )
    return ctx


def main(argv: Optional[Sequence[str]] = None) -> int:
    ctx = parse_args(argv)
    configure_logging(ctx.verbose)
    app = TinyJam(ctx)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())

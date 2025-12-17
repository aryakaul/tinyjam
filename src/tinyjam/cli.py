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
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from loguru import logger
from tqdm import tqdm
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

__version__ = "1.2.0"

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
TIMESTAMP_RE = re.compile(r"^(.+?)\s\((\d{2}:\d{2})-(\d{2}:\d{2})\)$")
ENGLISH_PREFIXES = ("en", "eng")
SUB_LINE_RE = re.compile(r"^([A-Za-z0-9][\w\.-]*)\s")
PLAYLIST_ORDER_CHOICES = ("shuffle", "forward", "reverse")
CURATED_JAMLIST_URL = (
    "https://raw.githubusercontent.com/aryakaul/tinyjam/refs/heads/main/assets/aryapproved/masterlist"
)


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


def get_video_height(file_path: str) -> Optional[int]:
    """Get video height using ffprobe. Returns None if unavailable."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=height",
        "-of", "csv=p=0",
        file_path,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            return int(result.stdout.strip())
        except ValueError:
            pass
    return None


def trim_video_segment(
    input_path: str, start_time: str, end_time: str, output_path: str
) -> bool:
    """Use ffmpeg to extract a video segment. Returns True on success."""
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-ss", start_time,
        "-to", end_time,
        "-i", input_path,
        "-c", "copy",  # Copy streams without re-encoding (fast)
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0


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


def extract_max_height(entry: Dict[str, Any]) -> Optional[int]:
    """Extract the maximum available video height (resolution) from metadata."""
    max_height = None

    # Check direct height field
    if "height" in entry and isinstance(entry["height"], int):
        max_height = entry["height"]

    # Check formats for maximum height
    formats = entry.get("formats")
    if isinstance(formats, list):
        for fmt in formats:
            if isinstance(fmt, dict):
                height = fmt.get("height")
                if isinstance(height, int):
                    if max_height is None or height > max_height:
                        max_height = height

    return max_height


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


def fetch_curated_jamlist() -> Path:
    try:
        with urlopen(CURATED_JAMLIST_URL, timeout=30) as response:
            payload = response.read()
    except (HTTPError, URLError) as exc:
        logger.error("failed to download curated jam list | {}", exc)
        sys.exit(1)
    except OSError as exc:
        logger.error("network error loading curated jam list | {}", exc)
        sys.exit(1)

    if not payload:
        logger.error("curated jam list download returned no data")
        sys.exit(1)

    try:
        with tempfile.NamedTemporaryFile(
            "wb", delete=False, prefix="tinyjam-curated-", suffix=".txt"
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
    except OSError as exc:
        logger.error("failed to persist curated jam list | {}", exc)
        sys.exit(1)

    logger.info("using curated jam list from {}", CURATED_JAMLIST_URL)
    return temp_path


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
    playlist_order: str
    playlist: List[str] = field(default_factory=list)
    pending_downloads: List[str] = field(default_factory=list)
    timestamps: Dict[int, Tuple[str, str]] = field(default_factory=dict)
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
    max_height: Optional[int] = None


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

                # Check for timestamp pattern: "artist (MM:SS-MM:SS)"
                match = TIMESTAMP_RE.match(line)
                if match:
                    artist = match.group(1).strip()
                    start_time = match.group(2)
                    end_time = match.group(3)
                    # Store timestamp by playlist index (before appending)
                    playlist_index = len(items)
                    items.append(artist)
                    self.ctx.timestamps[playlist_index] = (start_time, end_time)
                    logger.debug("parsed timestamp | {} | index {} | {} - {}", artist, playlist_index, start_time, end_time)
                else:
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
                "force mode enabled | clearing cache files in {}",
                self.ctx.output,
            )
            if archive_path.exists():
                archive_path.unlink()
                logger.debug("removed download archive")
            if artist_cache.exists():
                artist_cache.unlink()
                logger.debug("removed artist cache")
            if failure_log.exists():
                failure_log.unlink()
                logger.debug("removed failure log")

        archive_path.touch(exist_ok=True)
        artist_cache.touch(exist_ok=True)

    def compute_pending_downloads(self) -> None:
        unique_artists = dedupe_preserve_order(self.ctx.playlist)
        self.ctx.pending_downloads = unique_artists.copy()

        cache_path = self.ctx.artist_cache
        self.ctx.cached_queries = {}
        self.ctx.cached_video_ids = set()

        # Load existing cache
        if cache_path and cache_path.exists():
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

        # Scan for existing files not in cache and add them
        if self.ctx.output.is_dir():
            valid_exts = {".mp4", ".webm", ".mkv", ".mov", ".m4v", ".mpg", ".mpeg"}
            for file_path in self.ctx.output.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_exts:
                    # Extract video ID from filename
                    match = re.search(r"_\[([A-Za-z0-9_-]+)\]_\(", file_path.name)
                    if match:
                        video_id = match.group(1)
                        # If this video ID isn't in cache, check if it matches any artist
                        if video_id not in self.ctx.cached_video_ids:
                            for artist in unique_artists:
                                # Check if this might be the file for this artist
                                # by doing a search to see if the video ID matches
                                if artist not in self.ctx.cached_queries:
                                    selection = self.select_video(artist)
                                    if selection and selection.video_id == video_id:
                                        logger.info("found existing download | {} | adding to cache", artist)
                                        self.mark_downloaded(artist, video_id)
                                        self.ctx.cached_queries[artist] = video_id
                                        self.ctx.cached_video_ids.add(video_id)
                                        break

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

    def _ordered_playlist(self, items: Sequence[str]) -> List[str]:
        order = self.ctx.playlist_order
        if order == "forward":
            return list(items)
        if order == "reverse":
            return list(reversed(items))
        shuffled = list(items)
        random.shuffle(shuffled)
        return shuffled

    def _ordered_playlist_with_indices(self) -> List[Tuple[int, str]]:
        """Return ordered playlist as (original_index, artist) pairs."""
        indexed = list(enumerate(self.ctx.playlist))
        order = self.ctx.playlist_order
        if order == "forward":
            return indexed
        if order == "reverse":
            return list(reversed(indexed))
        shuffled = indexed.copy()
        random.shuffle(shuffled)
        return shuffled

    def _video_id_from_path(self, path: str) -> Optional[str]:
        name = Path(path).name
        match = re.search(r"_\[([A-Za-z0-9_-]+)\]_\(", name)
        if match:
            return match.group(1)
        return None

    def _get_cached_video_id(self, artist: str) -> Optional[str]:
        """Look up video ID for artist with case-insensitive fallback."""
        # Try exact match first
        if artist in self.ctx.cached_queries:
            logger.debug("cache hit (exact) | {} -> {}", artist, self.ctx.cached_queries[artist])
            return self.ctx.cached_queries[artist]

        # Fall back to case-insensitive match
        artist_lower = artist.lower()
        for cached_artist, video_id in self.ctx.cached_queries.items():
            if cached_artist.lower() == artist_lower:
                logger.debug("cache hit (case-insensitive) | {} matched {} -> {}", artist, cached_artist, video_id)
                return video_id

        logger.debug("cache miss | {} not found", artist)
        return None

    def _artist_from_path(self, path: str) -> Optional[str]:
        """Find artist name for a file path by matching video ID."""
        video_id = self._video_id_from_path(path)
        if not video_id:
            return None
        # Reverse lookup: find artist whose cached video ID matches
        for artist, cached_id in self.ctx.cached_queries.items():
            if cached_id == video_id:
                return artist
        return None

    def _ordered_download_files(self, files: List[str]) -> Tuple[List[str], int, int]:
        playlist = self.ctx.playlist
        expected = len(playlist)
        order = self.ctx.playlist_order

        file_by_id: Dict[str, str] = {}
        for path in files:
            video_id = self._video_id_from_path(path)
            if video_id:
                file_by_id[video_id] = path

        matched: List[str] = []
        referenced: Set[str] = set()

        if playlist and file_by_id:
            iterator: Iterable[str] = (
                playlist if order != "reverse" else reversed(playlist)
            )
            for artist in iterator:
                video_id = self._get_cached_video_id(artist)
                if not video_id:
                    continue
                file_path = file_by_id.get(video_id)
                if not file_path or file_path in referenced:
                    continue
                matched.append(file_path)
                referenced.add(file_path)

            if matched:
                return matched, len(matched), expected

        if order == "reverse":
            return list(reversed(files)), len(matched), expected
        return files, len(matched), expected

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
            max_height=extract_max_height(entry),
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

        # Warn about low quality videos
        if selection.max_height is not None:
            if selection.max_height < 720:
                logger.warning(
                    "low quality video | {} | max resolution: {}p (below 720p)",
                    artist,
                    selection.max_height,
                )
            elif selection.max_height < 1080:
                logger.debug(
                    "video quality | {} | max resolution: {}p",
                    artist,
                    selection.max_height,
                )
            else:
                logger.debug(
                    "video quality | {} | max resolution: {}p",
                    artist,
                    selection.max_height,
                )
        else:
            logger.debug("video quality | {} | resolution unknown", artist)

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

        playlist = self._ordered_playlist_with_indices()

        color_flag = ["--saturation=20"] if self.ctx.color else ["--saturation=-100"]
        sub_flags = [
            "--sub-auto=fuzzy",
            f"--slang={self.ctx.subtitle_lang}",
        ]
        base_cmd = ["mpv", *sub_flags, *color_flag, *STANDARD_OPS]

        for orig_idx, artist in playlist:
            cmd = base_cmd.copy()

            # Check if this original playlist position has a timestamp
            if orig_idx in self.ctx.timestamps:
                start_time, end_time = self.ctx.timestamps[orig_idx]
                logger.info("streaming | {} | {} - {}", artist, start_time, end_time)

                if not self.ctx.dry_run:
                    selection = self.select_video(artist)
                    if not selection:
                        logger.warning("could not resolve video for timestamped entry | {}", artist)
                        continue
                    video_url = selection.url
                else:
                    video_url = f"https://www.youtube.com/watch?v=<resolved-id>"

                cmd.extend([f"--start={start_time}", f"--end={end_time}", video_url])
            else:
                # No timestamp: use ytdl search protocol for simplicity
                query = f"ytdl://ytsearch1:npr tiny desk {artist}"
                cmd.append(query)
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

        # Build a map of artist -> file path
        # Also build video_id -> file path for more reliable lookups
        artist_to_file: Dict[str, str] = {}
        video_id_to_file: Dict[str, str] = {}

        for file_path in files:
            video_id = self._video_id_from_path(file_path)
            if video_id:
                video_id_to_file[video_id] = file_path
            artist = self._artist_from_path(file_path)
            if artist and artist not in artist_to_file:
                artist_to_file[artist] = file_path

        # Process playlist in order, creating trimmed segments as needed
        playback_files = []
        temp_dir = None
        has_timestamps = bool(self.ctx.timestamps)
        matched_count = 0

        playlist = self._ordered_playlist_with_indices()

        # First pass: check what files we'll actually play
        for orig_idx, artist in playlist:
            # Try to get video ID for this artist
            video_id = self._get_cached_video_id(artist)
            if video_id and video_id in video_id_to_file:
                matched_count += 1

        # Handle no matches case
        if matched_count == 0:
            logger.warning(
                "no downloaded files matched jam list | use -n to stream or download videos first"
            )
            logger.info("skipping playback | no matching files in {}", self.ctx.output)
            return

        if matched_count < len(self.ctx.playlist):
            missing_count = len(self.ctx.playlist) - matched_count
            logger.info(
                "playlist filter | found {}/{} jams | {} will be played, {} not downloaded",
                matched_count,
                len(self.ctx.playlist),
                matched_count,
                missing_count,
            )

        # Create temp directory only if we have timestamps AND matched files
        if has_timestamps and not self.ctx.dry_run:
            temp_dir = tempfile.mkdtemp(prefix="tinyjam-segments-")
            logger.debug("created temp directory for segments | {}", temp_dir)

        # Second pass: build playback list
        low_quality_count = 0
        for orig_idx, artist in playlist:
            # Look up file by video ID (most reliable method)
            video_id = self._get_cached_video_id(artist)
            file_path = None
            if video_id:
                file_path = video_id_to_file.get(video_id)

            if not file_path:
                logger.debug("skipping | {} | not downloaded", artist)
                continue

            # Check quality of this video (only for videos that will be played)
            if not self.ctx.dry_run and shutil.which("ffprobe"):
                height = get_video_height(file_path)
                if height is not None and height < 720:
                    file_name = Path(file_path).name
                    low_quality_count += 1
                    logger.warning(
                        "low quality video in playlist | {} | {}p | file: {}",
                        artist,
                        height,
                        file_name,
                    )

            # Check if this playlist entry has a timestamp
            if orig_idx in self.ctx.timestamps:
                start_time, end_time = self.ctx.timestamps[orig_idx]
                logger.info("trimming segment | {} | {} - {}", Path(file_path).name, start_time, end_time)

                if self.ctx.dry_run:
                    trimmed_path = f"{temp_dir or '/tmp'}/trimmed_{orig_idx}_{Path(file_path).name}"
                    logger.info(
                        "dry run | {}",
                        shlex_join([
                            "ffmpeg", "-y", "-ss", start_time, "-to", end_time,
                            "-i", file_path, "-c", "copy", "-avoid_negative_ts",
                            "make_zero", trimmed_path
                        ])
                    )
                    playback_files.append(trimmed_path)
                else:
                    # Create trimmed segment with unique name per index
                    file_ext = Path(file_path).suffix
                    trimmed_path = str(Path(temp_dir) / f"trimmed_{orig_idx}_{start_time.replace(':', '')}-{end_time.replace(':', '')}{file_ext}")

                    if trim_video_segment(file_path, start_time, end_time, trimmed_path):
                        playback_files.append(trimmed_path)
                    else:
                        logger.warning("failed to trim segment | {} | using full video", file_path)
                        playback_files.append(file_path)
            else:
                playback_files.append(file_path)

        # Report low quality videos summary
        if low_quality_count > 0:
            logger.info(
                "playlist contains {} low quality video(s) below 720p | delete and re-run to download higher quality",
                low_quality_count,
            )

        shuffle_flag = ["--shuffle"] if self.ctx.playlist_order == "shuffle" else []
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
            *shuffle_flag,
            *playback_files,
        ]

        if self.ctx.dry_run:
            logger.info("dry run | {}", shlex_join(cmd))
            return

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.warning("mpv exited with error | {}", self.ctx.output)

        # Cleanup temp directory
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
                logger.debug("cleaned up temp directory | {}", temp_dir)
            except OSError as exc:
                logger.warning("failed to cleanup temp directory | {} | {}", temp_dir, exc)

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
            # Check for ffmpeg if timestamps are present
            if self.ctx.timestamps:
                require_cmd("ffmpeg")
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


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    require_list: bool = True,
) -> TinyJamContext:
    parser = argparse.ArgumentParser(
        description="Jam to tiny desks with tinyjam (Python edition)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"tinyjam {__version__}",
    )
    parser.add_argument(
        "-l",
        "--list",
        required=require_list,
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
        "-p",
        "--playlist-order",
        choices=PLAYLIST_ORDER_CHOICES,
        default="shuffle",
        help=(
            "playback order: shuffle (default), jam list order (forward), "
            "or reverse jam list order"
        ),
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
        playlist_order=args.playlist_order,
    )
    return ctx


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    use_curated_default = len(args_list) == 0
    ctx = parse_args(args_list, require_list=not use_curated_default)
    configure_logging(ctx.verbose)
    if use_curated_default:
        ctx.jamlist = fetch_curated_jamlist()
        ctx.nodownload = True
        logger.info("no options provided | defaulting to streaming mode")
    app = TinyJam(ctx)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())

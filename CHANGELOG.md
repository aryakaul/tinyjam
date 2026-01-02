# Changelog

All notable changes to tinyjam will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - Unreleased

### Added
- Mixtape mode: specify video segments using `artist (MM:SS-MM:SS)` syntax in jam lists
- `--version` flag to display current version
- Auto-detection to rebuild cache from existing downloaded files
- Case-insensitive artist name matching for cache lookups
- `-q/--quality-check` flag for opt-in quality warnings
  - Checks video resolution during download and playback
  - Warns about low-resolution videos (below 720p)
  - Uses ffprobe to detect resolution efficiently
  - Provides actionable guidance: delete low-quality files and re-run to download higher quality
  - Quality checks are now opt-in (previously always-on) to improve performance
- `-a/--audio-out DIR` flag for audio extraction
  - Extracts audio from downloaded videos to numbered mp3 files
  - Files numbered based on playlist order (forward/reverse/shuffle): `01-Artist.mp3`, `02-Artist.mp3`, etc.
  - Supports mixtape mode (extracts only timestamped segments when specified)
  - Skips video playback when audio extraction is enabled
  - Uses ffmpeg with high-quality VBR mp3 encoding (~190kbps)
  - Automatic loudness normalization (EBU R128 standard, -16 LUFS target) ensures consistent volume across tracks
  - Incompatible with `--nodownload` (requires downloaded videos)

### Changed
- Local playback now filters to the jam list (ignores stray downloads)
- `--force` flag no longer deletes video files (only clears cache)
- Improved logging clarity for cache hits and playlist filtering
- Logging now shows exact vs case-insensitive cache matches in verbose mode

### Fixed
- Shuffle mode now shuffles playlist once and maintains that order across loops (no more back-to-back duplicates)
- Playlist playback now finds files even when artist name casing differs between jam lists and cache
- Streaming mode URL resolution for timestamped entries
- Downloaded mode now creates separate trimmed files per playlist entry (supports duplicate artists with different timestamps)

## [1.1.0] - 2025-11-19

### Added
- `-p/--playlist-order` flag to honor jam list order (forward, reverse, or shuffle)

## [1.0.0] - 2025-11-13

### Added
- Initial Python package release
- Packaged CLI with proper entry point
- Homebrew formula for easy installation

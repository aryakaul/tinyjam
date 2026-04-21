# Version Bump Checklist

This document outlines all steps required to release a new version of tinyjam.

## Current State
- Homebrew: v1.2.0 (aryakaul/formulae/tinyjam)
- PyPI: v1.2.0
- Next release: v1.3.0

## Pre-Release Checklist

### 1. Update Version Numbers
- [ ] `pyproject.toml` - Update `version = "X.Y.Z"` 
- [ ] `src/tinyjam/__init__.py` - Update `__version__ = "X.Y.Z"` 
- [ ] `CHANGELOG.md` - Change `[X.Y.Z] - Unreleased` to `[X.Y.Z] - YYYY-MM-DD` with today's date

### 2. Git Operations
- [ ] Ensure all changes are committed on `dev` branch
- [ ] Review git status: `git status` (should be clean)
- [ ] Review recent commits: `git log --oneline -5`

### 3. Build & Test Python Package
- [ ] Clean previous builds: `rm -rf build/ dist/ src/*.egg-info/`
- [ ] Build package: `python -m build`
- [ ] Check package: `python -m twine check dist/*`
- [ ] Verify package contents: `tar -tzf dist/tinyjam-X.Y.Z.tar.gz | head -20`

### 4. Create Git Tag
- [ ] Create annotated tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Verify tag: `git tag -l -n1 vX.Y.Z`

### 5. Push to GitHub
- [ ] Push dev branch: `git push origin dev`
- [ ] Push tag: `git push origin vX.Y.Z`
- [ ] Verify on GitHub: https://github.com/aryakaul/nprtinyjam/releases

## Release to PyPI

### 6. Upload to PyPI
- [ ] Upload to PyPI: `python -m twine upload dist/*`
- [ ] Verify on PyPI: https://pypi.org/project/tinyjam/
- [ ] Test installation: `pip install --upgrade tinyjam`
- [ ] Verify installed version: `tinyjam --version` or `pip show tinyjam`

## Release to Homebrew

### 7. Update Homebrew Formula
The Homebrew formula lives in a separate repository: `aryakaul/homebrew-formulae`

- [ ] Clone/update formulae repo: `git clone git@github.com:aryakaul/homebrew-formulae.git` (or `git pull` if already cloned)
- [ ] Edit `Formula/tinyjam.rb`:
  - [ ] Update `url` to point to new PyPI version tarball
  - [ ] Download new tarball: `curl -L https://files.pythonhosted.org/packages/.../tinyjam-X.Y.Z.tar.gz -o /tmp/tinyjam.tar.gz`
  - [ ] Calculate SHA256: `shasum -a 256 /tmp/tinyjam.tar.gz`
  - [ ] Update `sha256` in formula with new hash
  - [ ] Update `version "X.Y.Z"` if not auto-detected
  - [ ] Update any dependency changes if applicable
- [ ] Test formula locally: `brew install --build-from-source ./Formula/tinyjam.rb`
- [ ] Verify installation: `tinyjam --version`
- [ ] Commit changes: `git add Formula/tinyjam.rb && git commit -m "Bump tinyjam to vX.Y.Z"`
- [ ] Push to GitHub: `git push origin main`
- [ ] Wait for GitHub to update tap (~5 minutes)
- [ ] Test tap installation: `brew upgrade tinyjam` or `brew reinstall tinyjam`

## Post-Release Checklist

### 8. Merge to Main
- [ ] Switch to main: `git checkout main`
- [ ] Pull latest: `git pull origin main`
- [ ] Merge dev: `git merge dev`
- [ ] Push to GitHub: `git push origin main`

### 9. Prepare for Next Release
- [ ] Switch back to dev: `git checkout dev`
- [ ] Add new section to `CHANGELOG.md`:
  ```markdown
  ## [X.Y.Z+1] - Unreleased

  ### Added

  ### Changed

  ### Fixed
  ```
- [ ] Commit changelog: `git add CHANGELOG.md && git commit -m "Prepare changelog for next release"`
- [ ] Push to GitHub: `git push origin dev`

### 10. Verify Release
- [ ] PyPI shows correct version: https://pypi.org/project/tinyjam/
- [ ] Homebrew shows correct version: `brew info tinyjam`
- [ ] GitHub release tag exists: https://github.com/aryakaul/nprtinyjam/releases
- [ ] Both installation methods work:
  - [ ] `pip install tinyjam` installs vX.Y.Z
  - [ ] `brew install aryakaul/formulae/tinyjam` installs vX.Y.Z

## Troubleshooting

**Build fails:**
- Check `pyproject.toml` syntax
- Ensure all dependencies are listed correctly
- Verify `src/tinyjam/__init__.py` has correct imports

**Twine upload fails:**
- Check PyPI credentials: `~/.pypirc`
- Ensure version doesn't already exist on PyPI
- Verify package builds successfully: `python -m twine check dist/*`

**Homebrew formula fails:**
- Verify SHA256 hash matches downloaded tarball
- Check that PyPI tarball is accessible
- Ensure all Python dependencies are listed as resources
- Test formula syntax: `brew install --build-from-source ./Formula/tinyjam.rb`

**Git tag conflicts:**
- Check existing tags: `git tag -l`
- Delete local tag if needed: `git tag -d vX.Y.Z`
- Delete remote tag if needed: `git push origin :refs/tags/vX.Y.Z` (use with caution!)

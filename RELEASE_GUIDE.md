# Release Guide

Complete guide for releasing muvera to PyPI and creating GitHub releases.

## 🔧 One-time Setup

### 1. PyPI Account & OIDC Configuration

1. Create account at https://pypi.org
2. Go to Account Settings → Publishing → "Add a new pending publisher"
3. Fill in the form:
   ```
   PyPI Project Name: muvera
   Owner: craftsangjae
   Repository name: muvera-python
   Workflow name: publish.yml
   Environment name: (leave empty)
   ```
4. Click "Add"

**Note**: The project name `muvera` will be reserved. First release will claim it.

### 2. TestPyPI (Optional, for testing)

For testing releases before production:

1. Create account at https://test.pypi.org
2. Configure trusted publisher (same as above)
3. Update workflow to use TestPyPI temporarily

---

## 🚀 Release Process

### Step 1: Merge PR to main

```bash
# After PR approval, merge via GitHub UI or:
gh pr merge <PR-number> --squash
# or
gh pr merge <PR-number> --merge
```

### Step 2: Update local main branch

```bash
git checkout main
git pull origin main
```

### Step 3: Update version

Edit `pyproject.toml`:
```toml
[project]
version = "0.1.0"  # Update this
```

Also update `muvera/__init__.py`:
```python
__version__ = "0.1.0"  # Keep in sync
```

### Step 4: Pre-release checklist

```bash
# Run all checks locally
pytest tests/
ruff check .
ruff format --check .
mypy muvera

# Test build locally
pip install build
python -m build
twine check dist/*

# Clean up
rm -rf dist/ build/ *.egg-info
```

### Step 5: Commit version bump

```bash
git add pyproject.toml muvera/__init__.py
git commit -m "Bump version to 0.1.0"
git push origin main
```

### Step 6: Create and push tag

```bash
# Create annotated tag
git tag -a v0.1.0 -m "Release v0.1.0

- Add examples and benchmarks
- Add real-world ColBERT tests
- Comprehensive documentation
- CI/CD pipeline
"

# Push tag (this triggers PyPI deployment!)
git push origin v0.1.0
```

### Step 7: Monitor GitHub Actions

1. Go to https://github.com/craftsangjae/muvera-python/actions
2. Watch the "Publish to PyPI" workflow
3. Verify:
   - ✅ Tests pass
   - ✅ Version matches tag
   - ✅ Build succeeds
   - ✅ PyPI upload succeeds

### Step 8: Create GitHub Release

#### Option A: Manual (Recommended for first release)

1. Go to https://github.com/craftsangjae/muvera-python/releases/new
2. Choose tag: `v0.1.0`
3. Release title: `v0.1.0`
4. Description:
   ```markdown
   ## MuVERA v0.1.0 - Initial Release

   First public release of MuVERA Python implementation.

   ### ✨ Features
   - Clean, simple API: `encode_documents()` and `encode_queries()`
   - Support for variable-length batches (`list[np.ndarray]`)
   - Multiple projection types: identity, AMS sketch
   - Optional Count Sketch final projection
   - Validated against reference implementation

   ### 📚 Documentation
   - Comprehensive examples (`basic_usage.py`, `colbert_nanobeir.py`)
   - Full test suite with real ColBERT embeddings
   - Development guide in CLAUDE.md

   ### 📦 Installation
   ```bash
   pip install muvera
   ```

   ### 🔗 Links
   - PyPI: https://pypi.org/project/muvera/
   - Documentation: https://github.com/craftsangjae/muvera-python
   - Paper: https://arxiv.org/abs/2405.19504
   ```

5. Attach artifacts (optional):
   - Download from Actions → Build artifacts
   - Upload `.whl` and `.tar.gz` files

6. Check "Set as the latest release"
7. Click "Publish release"

#### Option B: Automated (Future improvement)

Add to `.github/workflows/publish.yml`:

```yaml
- name: Create GitHub Release
  uses: softprops/action-gh-release@v1
  with:
    files: dist/*
    generate_release_notes: true
```

### Step 9: Verify deployment

```bash
# Wait ~2 minutes for PyPI to process

# Check PyPI page
open https://pypi.org/project/muvera/

# Test installation in fresh environment
python -m venv test_env
source test_env/bin/activate
pip install muvera

# Quick test
python -c "from muvera import Muvera; print(Muvera.__doc__)"
```

### Step 10: Announce (Optional)

- Update README.md with installation instructions
- Tweet/post on relevant communities
- Update project homepage

---

## 🔄 Subsequent Releases

For patch/minor releases, follow the same process:

```bash
# 1. Update version
vim pyproject.toml  # 0.1.0 -> 0.1.1

# 2. Commit
git commit -am "Bump version to 0.1.1"
git push

# 3. Tag and push
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1

# 4. Create GitHub release
gh release create v0.1.1 --generate-notes
```

---

## 🐛 Troubleshooting

### PyPI upload fails: "File already exists"

**Cause**: Version already published (PyPI doesn't allow overwrites)

**Solution**: Increment version and create new tag

### PyPI upload fails: "Invalid authentication"

**Cause**: OIDC trusted publisher not configured

**Solution**:
1. Check PyPI trusted publisher settings
2. Ensure repository, workflow name match exactly
3. Verify workflow has `id-token: write` permission

### Tag version doesn't match package version

**Cause**: Version verification step in workflow fails

**Solution**:
```bash
# Fix version in pyproject.toml
git add pyproject.toml
git commit -m "Fix version"
git push

# Delete and recreate tag
git tag -d v0.1.0
git push origin :v0.1.0
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

### Tests fail in workflow

**Cause**: Code issues or environment differences

**Solution**: Fix issues, commit, and retag

---

## 📋 Release Checklist

Before tagging:

- [ ] All tests pass locally (`pytest tests/`)
- [ ] Code formatted (`ruff format .`)
- [ ] No lint errors (`ruff check .`)
- [ ] Type checking passes (`mypy muvera`)
- [ ] Version updated in `pyproject.toml`
- [ ] Version updated in `muvera/__init__.py`
- [ ] CHANGELOG updated (if exists)
- [ ] Examples tested
- [ ] Documentation reviewed

After tagging:

- [ ] GitHub Actions workflow succeeds
- [ ] Package appears on PyPI
- [ ] GitHub Release created
- [ ] Installation tested (`pip install muvera`)
- [ ] README updated

---

## 📚 Semantic Versioning

Follow [SemVer](https://semver.org/):

- `MAJOR.MINOR.PATCH` (e.g., `1.2.3`)

**MAJOR** (1.0.0 → 2.0.0):
- Incompatible API changes
- Breaking changes

**MINOR** (1.0.0 → 1.1.0):
- New features (backward compatible)
- New functionality

**PATCH** (1.0.0 → 1.0.1):
- Bug fixes
- Documentation updates
- Internal improvements

**Examples**:
- Bug fix: `0.1.0` → `0.1.1`
- New feature: `0.1.1` → `0.2.0`
- Breaking change: `0.2.0` → `1.0.0`

---

## 🎯 Quick Reference

```bash
# Full release in one script
VERSION="0.1.0"

# Update version
sed -i "s/version = .*/version = \"$VERSION\"/" pyproject.toml
sed -i "s/__version__ = .*/__version__ = \"$VERSION\"/" muvera/__init__.py

# Run checks
pytest tests/ && ruff check . && ruff format --check . && mypy muvera

# Commit and tag
git add pyproject.toml muvera/__init__.py
git commit -m "Bump version to $VERSION"
git push origin main
git tag -a "v$VERSION" -m "Release v$VERSION"
git push origin "v$VERSION"

# Create GitHub release
gh release create "v$VERSION" --generate-notes

# Verify
pip install --upgrade muvera
```

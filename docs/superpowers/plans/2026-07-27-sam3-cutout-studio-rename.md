# SAM3 Cutout Studio Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename every public and local product surface to SAM3 Cutout Studio with the identical `techfreakworm/SAM3-cutout-studio` GitHub and Hugging Face identifiers.

**Architecture:** Treat the product title, repository identifier, Python distribution name, and checkout directory as distinct named surfaces governed by one canonical matrix. Preserve the Python import package and finalized reference-workflow filenames because they are implementation/provenance identifiers rather than branding.

**Tech Stack:** GitHub CLI, Git, Hugging Face Hub metadata, Python packaging, Gradio, pytest, Ruff

---

### Task 1: Rename the public repository and Hinode checkout

**Files:**
- Move checkout: `/home/wakeuser/Projects/llm/sam3-pro-vitmatte-bg-removal` to `/home/wakeuser/Projects/llm/SAM3-cutout-studio`
- Update Git remote: `origin`

- [ ] **Step 1: Rename the GitHub repository in place**

Run from the authenticated control plane:

```bash
gh repo rename SAM3-cutout-studio --repo techfreakworm/sam3-pro-vitmatte-bg-removal --yes
```

Expected: the public repository resolves as `techfreakworm/SAM3-cutout-studio` and the old URL redirects.

- [ ] **Step 2: Rename the Hinode checkout directory**

```bash
ssh wakeuser@hinode 'mv "$HOME/Projects/llm/sam3-pro-vitmatte-bg-removal" "$HOME/Projects/llm/SAM3-cutout-studio"'
```

Expected: only `/home/wakeuser/Projects/llm/SAM3-cutout-studio` exists.

- [ ] **Step 3: Point the checkout at the canonical GitHub identifier**

```bash
git remote set-url origin git@github.com:techfreakworm/SAM3-cutout-studio.git
git remote -v
```

Expected: fetch and push both use `techfreakworm/SAM3-cutout-studio.git`.

### Task 2: Update the branding contract test first

**Files:**
- Modify: `tests/test_ui.py`
- Modify: `src/sam3_matting/ui.py`

- [ ] **Step 1: Change the UI contract assertion to the approved title**

```python
assert "SAM3 Cutout Studio" in serialized_config
assert "SAM3 Pro Matte Studio" not in serialized_config
```

- [ ] **Step 2: Run the targeted test and verify RED**

```bash
PATH="$HOME/.local/bin:$PATH" .venv/bin/python -m pytest tests/test_ui.py -q
```

Expected: FAIL while the partial UI still contains the old title or does not return a Blocks instance.

- [ ] **Step 3: Update UI copy and complete the existing event binding**

Use `SAM3 Cutout Studio` for the product title and `Background, gone. Detail, intact.` for the concise product promise. Keep the approved ten-input/four-output handler contract unchanged.

- [ ] **Step 4: Run the targeted test and verify GREEN**

```bash
PATH="$HOME/.local/bin:$PATH" .venv/bin/python -m pytest tests/test_ui.py -q
```

Expected: all UI tests pass.

### Task 3: Update repository and package metadata

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `docs/DESIGN.md`
- Modify: `THIRD_PARTY_NOTICES.md` only if it contains the old public identity

- [ ] **Step 1: Update exact metadata values**

Set the README Space title to `SAM3 Cutout Studio`, the H1 to `SAM3 Cutout Studio`, and `[project].name` to `sam3-cutout-studio`. Preserve `src/sam3_matting` as the wheel package.

- [ ] **Step 2: Scan for stale branding**

```bash
git grep -n -E 'SAM3 Pro Matte Studio|sam3-pro-vitmatte-bg-removal|sam3-matte-studio'
```

Expected: no matches outside the approved naming design/rename-plan historical migration text.

### Task 4: Verify, commit, and push the rename

**Files:**
- Modify: all files listed above
- Add: `docs/superpowers/specs/2026-07-27-sam3-cutout-studio-naming-design.md`
- Add: `docs/superpowers/plans/2026-07-27-sam3-cutout-studio-rename.md`

- [ ] **Step 1: Run repository verification**

```bash
PATH="$HOME/.local/bin:$PATH" .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
git diff --check
```

Expected: all implemented tests pass, Ruff reports no errors, formatting is clean, and Git reports no whitespace errors.

- [ ] **Step 2: Commit on Hinode**

```bash
git add README.md pyproject.toml docs src/sam3_matting/ui.py tests/test_ui.py assets
git commit -m "refactor: rename project to SAM3 Cutout Studio"
```

- [ ] **Step 3: Push from Hinode**

```bash
git push origin main
```

Expected: GitHub `main` contains the rename commit under `techfreakworm/SAM3-cutout-studio`.

### Task 5: Reserve the matching Hugging Face identity

**Files:**
- Deployment target only: `techfreakworm/SAM3-cutout-studio`

- [ ] **Step 1: Create the public Gradio Space only after CUDA end-to-end verification**

Create `techfreakworm/SAM3-cutout-studio` with SDK `gradio` and ZeroGPU hardware, then configure GitHub as the source of truth.

- [ ] **Step 2: Verify identity consistency**

Confirm the Space URL, README title, Gradio header, browser title, and GitHub repository all display `SAM3 Cutout Studio` or the exact `SAM3-cutout-studio` identifier.

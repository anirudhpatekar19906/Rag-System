"""
github_loader.py
----------------
Clones a GitHub repository (or reads a local path) and extracts the text
content of all source-code / text files.

Returns a list of dicts:
    [{"file": "src/main.py", "text": "...", "source": "gh:<repo>:src/main.py"}, ...]
"""

import os
import shutil
import tempfile
from pathlib import Path

import git  # GitPython

# File extensions we consider "readable" source / text files
READABLE_EXTENSIONS: set[str] = {
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".r", ".m", ".sh", ".bash", ".zsh", ".ps1",
    # Config / data
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    # Docs / markup
    ".md", ".rst", ".txt", ".html", ".htm", ".xml", ".csv",
    # Notebooks (raw JSON)
    ".ipynb",
}

# Directories to always skip
SKIP_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".next", ".nuxt", "coverage",
}

MAX_FILE_BYTES = 512_000  # skip files larger than ~500 KB


def _is_readable(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in READABLE_EXTENSIONS
        and path.stat().st_size <= MAX_FILE_BYTES
    )


def _should_skip_dir(directory: Path) -> bool:
    return any(part in SKIP_DIRS for part in directory.parts)


def _read_files(root: Path, repo_label: str) -> list[dict]:
    """Walk the directory tree and collect readable file contents."""
    results = []
    for path in sorted(root.rglob("*")):
        # Skip unwanted directories
        rel = path.relative_to(root)
        if _should_skip_dir(rel):
            continue
        if not _is_readable(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                rel_str = str(rel)
                results.append({
                    "file": rel_str,
                    "text": text,
                    "source": f"gh:{repo_label}:{rel_str}",
                })
        except Exception as exc:
            print(f"[github_loader] Warning: could not read {path}: {exc}")

    return results


def load_github(repo_url_or_path: str, branch: str | None = None) -> list[dict]:
    """
    Load all source files from a GitHub repo URL or a local directory path.

    Args:
        repo_url_or_path: GitHub HTTPS URL (e.g. "https://github.com/user/repo")
                          or a local path to an existing repo clone.
        branch:           Branch / tag to checkout. Defaults to the repo's default branch.

    Returns:
        List of {"file": str, "text": str, "source": str} dicts.
    """
    is_url = repo_url_or_path.startswith(("http://", "https://", "git@"))
    tmp_dir: str | None = None

    try:
        if is_url:
            tmp_dir = tempfile.mkdtemp(prefix="rag_gh_")
            print(f"[github_loader] Cloning {repo_url_or_path} …")
            clone_kwargs: dict = {"to_path": tmp_dir, "depth": 1}
            if branch:
                clone_kwargs["branch"] = branch
            git.Repo.clone_from(repo_url_or_path, **clone_kwargs)
            root = Path(tmp_dir)
            label = repo_url_or_path.rstrip("/").split("/")[-1]
        else:
            root = Path(repo_url_or_path).resolve()
            if not root.exists():
                raise FileNotFoundError(f"Path does not exist: {root}")
            label = root.name

        files = _read_files(root, label)
        print(f"[github_loader] Loaded {len(files)} files from '{label}'")
        return files

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def load_github_as_text(repo_url_or_path: str, branch: str | None = None) -> str:
    """
    Convenience wrapper — returns all files joined as one big string,
    each preceded by its relative path as a header.
    """
    files = load_github(repo_url_or_path, branch)
    parts = [f"### {f['file']}\n{f['text']}" for f in files]
    return "\n\n".join(parts)


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python github_loader.py <github_url_or_local_path>")
        sys.exit(1)

    items = load_github(sys.argv[1])
    for item in items[:5]:
        print(f"\n--- {item['file']} ---")
        print(item["text"][:300])
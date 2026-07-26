"""Write a checksum and provenance manifest for the public local inputs."""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifact" / "builds"
INPUTS = {"moralchoice": ROOT / "analysis" / "moralchoice", "prompteval": ROOT / "analysis" / "prompteval", "mtbench": ROOT / "analysis" / "mtbench"}

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for source, directory in INPUTS.items():
        if not directory.exists(): raise FileNotFoundError(directory)
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            files.append({"source": source, "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "repository_commit": git_head(ROOT),
        "source_commits": {
            "moralchoice": git_head(ROOT / "external_sources" / "moralchoice"),
            "prompteval": "1639d5ea14c362f6964f260ae81bd903af760187",
            "mtbench": "6e465b26cb18b64e48b3858d54ac655736cf07b6",
        },
        "files": files,
    }
    out = OUT / "public_input_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)} with {len(files)} checksummed files")

if __name__ == "__main__": main()

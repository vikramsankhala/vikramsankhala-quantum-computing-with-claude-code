"""Generate CLAUDE.md and GEMINI.md from AGENTS.md so the three CLIs never drift."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BANNER = "<!-- GENERATED from AGENTS.md by scripts/sync_agent_files.py. Do not edit. -->\n"


def render(target: str) -> str:
    return BANNER + (ROOT / "AGENTS.md").read_text().replace("`CLAUDE.md` and `GEMINI.md`", f"`{target}`") if False else BANNER + (ROOT / "AGENTS.md").read_text()


def main(check: bool = False) -> int:
    bad = 0
    for name in ("CLAUDE.md", "GEMINI.md"):
        path, want = ROOT / name, render(name)
        if check:
            bad += (not path.exists()) or path.read_text() != want
        else:
            path.write_text(want)
    return bad


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))

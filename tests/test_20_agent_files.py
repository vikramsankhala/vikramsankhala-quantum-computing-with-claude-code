import json, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_generated_files_in_sync():
    assert subprocess.run([sys.executable, str(ROOT / "scripts/sync_agent_files.py"), "--check"]).returncode == 0


def test_mcp_configs_point_at_same_server():
    claude = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["qcwork"]
    gemini = json.loads((ROOT / ".gemini/settings.json").read_text())["mcpServers"]["qcwork"]
    codex = (ROOT / ".codex/config.toml").read_text()
    assert claude == gemini and claude["args"] == ["mcp_server/server.py"]
    assert "[mcp_servers.qcwork]" in codex and "mcp_server/server.py" in codex
    assert (ROOT / "mcp_server/server.py").exists()

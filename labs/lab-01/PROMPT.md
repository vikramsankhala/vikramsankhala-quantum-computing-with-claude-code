# Lab 01: Hilbert space from scratch

**Module:** `src/qcwork/hilbert.py`  **Tests:** `tests/`

## Task
Implement inner product, norm, Gram-Schmidt and expansion in a basis. Prove Cauchy-Schwarz numerically on 1,000 random pairs.

## Prompt (works in Claude Code, Codex CLI and Gemini CLI)
```
Read AGENTS.md first. Then: Implement inner product, norm, Gram-Schmidt and expansion in a basis. Prove Cauchy-Schwarz numerically on 1,000 random pairs.
Write the failing test first, then the implementation. Run `python -m pytest -q` and show me the output.
Do not change any test to make it pass.
```

## Verify
- [ ] `python -m pytest -q` is green.
- [ ] You can explain every equation in the docstring without looking at it.
- [ ] You ran the same prompt in a second CLI and compared the diffs.

## Run it in each CLI
| CLI | Start | Project context |
|---|---|---|
| Claude Code | `claude` | `CLAUDE.md`, `.mcp.json` |
| Codex CLI | `codex` (or `codex exec "<prompt>"`) | `AGENTS.md`, `.codex/config.toml` |
| Gemini CLI | `gemini` | `GEMINI.md`, `.gemini/settings.json` |

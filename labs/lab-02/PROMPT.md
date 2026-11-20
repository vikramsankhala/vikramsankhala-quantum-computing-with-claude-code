# Lab 02: Operators, measurement and the Schrödinger equation

**Module:** `src/qcwork/operators.py`  **Tests:** `tests/`

## Task
Implement Hermitian/unitary checks, the Born rule and U=exp(-iHt). Verify RK4 against the matrix exponential.

## Prompt (works in Claude Code, Codex CLI and Gemini CLI)
```
Read AGENTS.md first. Then: Implement Hermitian/unitary checks, the Born rule and U=exp(-iHt). Verify RK4 against the matrix exponential.
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

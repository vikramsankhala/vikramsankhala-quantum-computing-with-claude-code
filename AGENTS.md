# qcwork: instructions for AI coding agents

This is the single source of truth. `CLAUDE.md` and `GEMINI.md` are generated from it by
`python scripts/sync_agent_files.py`; do not edit them by hand.

## What this repository is
A tested NumPy reference implementation of quantum computing, from Hilbert space to
distributed circuits. Every chapter of the book has a module and tests here.

## Commands
- Install: `pip install -e ".[dev]"`
- Test everything: `python -m pytest -q`
- Test one module: `python -m pytest tests/test_11_contextuality.py -q`
- Sync agent files: `python scripts/sync_agent_files.py`

## Conventions
- Qubit 0 is the most significant bit (big-endian), as in Cirq and PennyLane. Qiskit is
  little-endian: reverse bit order when comparing.
- Physics units: hbar = 1.
- Every public function has a test. A new function without a test is not finished.
- Never hard-code a number in prose that a test or script did not produce.

## Definition of done
1. `python -m pytest -q` passes. 2. No new dependency without a reason in the PR.
3. Docstring states the equation or protocol implemented.

## Tools
An MCP server (`mcp_server/server.py`) exposes `run_circuit`, `bloch`, `estimate_resources`
and `magic_square`. Prefer calling these over re-deriving results by hand.

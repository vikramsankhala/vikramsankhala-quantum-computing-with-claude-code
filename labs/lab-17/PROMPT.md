# Lab 17: Entanglement and entanglement-based cryptography

**Module:** `src/qcwork/entanglement.py`  **Tests:** `tests/test_17_entanglement.py`

## Task
Compute the CHSH value from a Bell state and show the 2 sqrt(2) bound. Simulate E91 key distribution with and without an intercept-resend eavesdropper; show that CHSH falls below 2 when she listens. Implement superdense coding, entanglement swapping (all four outcomes) and GHZ secret sharing.

## Prompt (works in Claude Code, Codex CLI and Gemini CLI)
```
Read AGENTS.md first. Then: implement src/qcwork/entanglement.py with chsh_value, chsh_from_state, e91(eve=False/True),
superdense_encode_decode, entanglement_swap and ghz_secret_sharing. Write the failing tests first. The tests must assert independently
derived numbers: S = 2*sqrt(2) for a Bell state, S <= 2 for a product state, and a visibility threshold of 1/sqrt(2).
Run python -m pytest -q. Do not change a test to make it pass.
```

## Verify
- [ ] The CHSH bound in your test is derived by hand, not copied from the code's output.
- [ ] With the eavesdropper switched on, S < 2 and the key error rate is well above zero.
- [ ] You ran the same prompt in a second CLI and compared the diffs.

"""MCP server exposing qcwork tools to Claude Code, Codex CLI and Gemini CLI.

Run:  python mcp_server/server.py     (stdio transport)
"""
from mcp.server.fastmcp import FastMCP

from qcwork import tools

mcp = FastMCP("qcwork")


@mcp.tool()
def run_circuit(n: int, gates: list[dict], shots: int = 0, seed: int = 0) -> dict:
    """Simulate an n-qubit circuit. gates: [{"gate":"h","qubits":[0]}, {"gate":"rx","angle":0.5,"qubits":[1]}]."""
    return tools.run_circuit(n, gates, shots, seed)


@mcp.tool()
def bloch(theta: float, phi: float) -> dict:
    """Bloch vector of cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    return tools.bloch(theta, phi)


@mcp.tool()
def estimate_resources(logical_qubits: int, physical_error_rate: float, target_failure: float = 0.01,
                       rounds: float = 1e6, overhead: float = 1.5) -> dict:
    """Teaching-model surface-code resource estimate."""
    return tools.estimate_resources(logical_qubits, physical_error_rate, target_failure, rounds, overhead)


@mcp.tool()
def magic_square() -> dict:
    """Classical vs quantum value of the 3x3 magic-square game and the KS-Bell inequality."""
    return tools.magic_square()


@mcp.tool()
def e91(rounds: int = 20000, eve: bool = False, seed: int = 0) -> dict:
    """Simulate E91 entanglement-based key distribution; returns CHSH value S, key error rate and whether S > 2."""
    from qcwork import entanglement
    return entanglement.e91(rounds, eve, seed)


if __name__ == "__main__":
    mcp.run()

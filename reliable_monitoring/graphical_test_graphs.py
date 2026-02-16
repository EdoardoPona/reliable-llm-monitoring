"""Graph construction utilities for the graphical testing procedure.

Provides factory functions that build ``(weights, transitions)`` pairs
for common graph topologies.  Each factory returns plain NumPy arrays
and knows nothing about risks, thresholds, or other problem-specific
semantics.

References
----------
Bretz, Maurer, Brannath & Posch (2009), "A graphical approach to
sequentially rejective multiple test procedures."
"""

import numpy as np


def chain_graph(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Chain (fixed-sequence) graph.

    All weight starts at node 0 and propagates forward on rejection:
    ``w[0] = 1``, ``g[i, i+1] = 1`` for ``i < m-1``.

    This is the graph that makes :func:`graphical_testing` equivalent
    to :func:`fixed_sequence_testing`.
    """
    weights = np.zeros(m)
    weights[0] = 1.0
    transitions = np.zeros((m, m))
    for i in range(m - 1):
        transitions[i, i + 1] = 1.0
    return weights, transitions


def lattice_graph(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Lattice graph for 2-dimensional hypothesis grids.

    Nodes are arranged in an ``(n_rows × n_cols)`` grid and flattened
    row-major: node ``(r, c)`` maps to index ``r * n_cols + c``.

    Edges go from ``(r, c)`` to ``(r+1, c)`` (down) and ``(r, c+1)``
    (right).  When a node has two outgoing edges the transition weight
    is split equally (0.5 each); boundary nodes with a single neighbor
    get weight 1.0.

    All initial alpha-weight is placed at node ``(0, 0)``.
    """
    m = n_rows * n_cols
    weights = np.zeros(m)
    weights[0] = 1.0

    transitions = np.zeros((m, m))
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            neighbors = []
            if r + 1 < n_rows:
                neighbors.append((r + 1) * n_cols + c)
            if c + 1 < n_cols:
                neighbors.append(r * n_cols + (c + 1))
            if neighbors:
                share = 1.0 / len(neighbors)
                for nb in neighbors:
                    transitions[idx, nb] = share

    return weights, transitions


def row_chain_graph(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent chain per row with surplus transfer between rows.

    Each row receives equal initial weight (``1 / n_rows``) at its
    first node.  Within a row, hypotheses are tested sequentially
    (chain: ``g[r*n_cols+c, r*n_cols+c+1] = 1``).  If every hypothesis
    in row *r* is rejected, its surplus weight flows to the first
    hypothesis of row *r+1*.

    This answers questions like "for each row-level, what is the best
    column-level we can achieve?" — each row has its own reserved
    budget and cannot be starved by earlier rows.

    The *transpose* query ("for each column-level, what is the best
    row-level?") is obtained by swapping ``n_rows`` and ``n_cols``
    and arranging hypotheses accordingly.
    """
    m = n_rows * n_cols
    weights = np.zeros(m)
    transitions = np.zeros((m, m))

    for r in range(n_rows):
        # First node of each row gets an equal share
        weights[r * n_cols] = 1.0 / n_rows

        # Chain within the row
        for c in range(n_cols - 1):
            transitions[r * n_cols + c, r * n_cols + c + 1] = 1.0

        # Surplus transfer: end of row r → start of row r+1
        if r < n_rows - 1:
            transitions[r * n_cols + n_cols - 1, (r + 1) * n_cols] = 1.0

    return weights, transitions


def uniform_lattice_graph(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Lattice graph with uniform initial weights.

    Same edge structure as :func:`lattice_graph` but the alpha budget
    is spread equally across all ``m = n_rows × n_cols`` nodes.
    """
    _, transitions = lattice_graph(n_rows, n_cols)
    m = n_rows * n_cols
    weights = np.ones(m) / m
    return weights, transitions

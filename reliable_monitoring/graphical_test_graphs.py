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


def uniform_lattice_graph(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Lattice graph with uniform initial weights.

    Same edge structure as :func:`lattice_graph` but the alpha budget
    is spread equally across all ``m = n_rows × n_cols`` nodes.
    """
    _, transitions = lattice_graph(n_rows, n_cols)
    m = n_rows * n_cols
    weights = np.ones(m) / m
    return weights, transitions

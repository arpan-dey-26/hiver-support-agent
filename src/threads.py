"""Thread reconstruction — single source of truth for the rest of the pipeline.

Same verified logic as Phase 0's inspection script: root resolution via a sorted
index, then a compacted single-edge ancestor walk for EXACT depth.

Do not replace the walk with pointer-jumping (root = root[root]). That returns
ceil(log2(depth)) — a chain of 8 reports depth 3 — and multi-turn share is a
brand-selection input. See DECISIONS.md D4.
"""

from __future__ import annotations

import numpy as np


def build_parent_index(tweet_id: np.ndarray, parent_id: np.ndarray):
    """Map each row to its parent ROW INDEX. Rows with no parent, or whose parent
    is absent from the file (orphans), point at themselves and are roots."""
    n = len(tweet_id)
    if n == 0:
        empty = np.array([], dtype="int64")
        return empty, np.array([], dtype=bool), empty
    order = np.argsort(tweet_id, kind="stable")
    sorted_ids = tweet_id[order]
    pos = np.clip(np.searchsorted(sorted_ids, parent_id), 0, n - 1)
    resolvable = (parent_id >= 0) & (sorted_ids[pos] == parent_id)
    self_idx = np.arange(n)
    parent_idx = np.where(resolvable, order[pos], self_idx)
    return parent_idx, resolvable, self_idx


def reconstruct_threads(tweet_id: np.ndarray, parent_id: np.ndarray,
                        max_depth: int = 1000) -> dict:
    """Return conversation root index and exact depth for every row.

    Nodes still walking at `max_depth` are in a cycle; they are counted and their
    depth is set to -1 so a corrupt value can never pass for a real one.
    """
    n = len(tweet_id)
    if n == 0:
        return {"conversation_id": np.array([], dtype="int64"),
                "depth": np.array([], dtype="int32"), "stats": {}}

    parent_idx, resolvable, self_idx = build_parent_index(tweet_id, parent_id)

    depth = np.zeros(n, dtype="int32")
    root = self_idx.copy()

    alive = np.flatnonzero(parent_idx != self_idx)
    cur = parent_idx[alive]
    depth[alive] = 1
    root[alive] = cur

    steps = 1
    while alive.size and steps < max_depth:
        nxt = parent_idx[cur]
        moving = nxt != cur
        alive, cur = alive[moving], nxt[moving]
        if alive.size == 0:
            break
        depth[alive] += 1
        root[alive] = cur
        steps += 1

    n_cycle = int(alive.size)
    if n_cycle:
        depth[alive] = -1

    valid = depth[depth >= 0]
    sizes = np.bincount(root, minlength=n)
    conv_sizes = sizes[sizes > 0]
    n_replies = int((parent_id >= 0).sum())
    n_orphans = int(((parent_id >= 0) & ~resolvable).sum())

    return {
        "conversation_id": root,
        "depth": depth,
        "parent_idx": parent_idx,
        "stats": {
            "n_tweets": int(n),
            "n_with_parent_field": n_replies,
            "n_parent_resolvable": int(resolvable.sum()),
            "n_orphan_replies": n_orphans,
            "orphan_rate_of_replies": float(n_orphans / max(n_replies, 1)),
            "n_conversations": int((sizes > 0).sum()),
            "n_unconverged_cycles": n_cycle,
            "depth_max": int(valid.max()) if valid.size else 0,
            "depth_mean": float(valid.mean()) if valid.size else 0.0,
            "conv_size_mean": float(conv_sizes.mean()) if conv_sizes.size else 0.0,
            "conv_size_max": int(conv_sizes.max()) if conv_sizes.size else 0,
        },
    }

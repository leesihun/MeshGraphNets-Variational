"""DP min-max partitioner for assigning contiguous blocks to pipeline stages.

Given per-block memory estimates and a target number of stages K, find an
assignment of consecutive blocks to stages such that the max stage cost is
minimized. Classic O(L^2 * K) dynamic program.
"""

from __future__ import annotations

from typing import List, Sequence


def partition_stages(block_costs: Sequence[float], num_stages: int) -> List[List[int]]:
    """Partition L contiguous blocks into `num_stages` groups.

    Args:
        block_costs: cost (bytes / flops / whatever) for each of L blocks in order.
        num_stages: number of stages K. Must satisfy 1 <= K <= L.

    Returns:
        List of K lists of block indices. Indices within a stage are contiguous
        and cover [0..L-1] exactly once.
    """
    L = len(block_costs)
    K = int(num_stages)
    if K < 1:
        raise ValueError(f"num_stages must be >= 1, got {K}")
    if K > L:
        raise ValueError(f"num_stages ({K}) cannot exceed number of blocks ({L})")
    if K == 1:
        return [list(range(L))]
    if K == L:
        return [[i] for i in range(L)]

    costs = [float(c) for c in block_costs]
    prefix = [0.0] * (L + 1)
    for i in range(L):
        prefix[i + 1] = prefix[i] + costs[i]

    def segment_cost(i: int, j: int) -> float:
        return prefix[j + 1] - prefix[i]

    INF = float('inf')
    # dp[k][i] = min over splits of the max-stage cost when assigning blocks [0..i] to k stages
    dp = [[INF] * L for _ in range(K + 1)]
    cut = [[0] * L for _ in range(K + 1)]

    for i in range(L):
        dp[1][i] = segment_cost(0, i)
        cut[1][i] = 0

    for k in range(2, K + 1):
        for i in range(k - 1, L):
            best = INF
            best_j = k - 2
            for j in range(k - 2, i):
                candidate = max(dp[k - 1][j], segment_cost(j + 1, i))
                if candidate < best:
                    best = candidate
                    best_j = j
            dp[k][i] = best
            cut[k][i] = best_j

    stages: List[List[int]] = []
    end = L - 1
    for k in range(K, 0, -1):
        start = cut[k][end] + 1 if k > 1 else 0
        stages.append(list(range(start, end + 1)))
        end = cut[k][end]
    stages.reverse()
    return stages


def partition_summary(block_costs: Sequence[float], assignment: Sequence[Sequence[int]]) -> str:
    """One-line summary of per-stage cost for logging."""
    parts = []
    for s, blocks in enumerate(assignment):
        s_cost = sum(block_costs[b] for b in blocks)
        idx_str = f"{blocks[0]}-{blocks[-1]}" if len(blocks) > 1 else f"{blocks[0]}"
        parts.append(f"stage{s}[{idx_str}]={s_cost / 1e6:.1f}MB")
    return ' | '.join(parts)

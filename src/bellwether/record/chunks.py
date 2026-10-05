"""Chunk plans: how an output token stream is cut into engine chunks for a streaming replay.

A plan is a list of chunk lengths in tokens that sums to the output length, or ``None`` for the two
plans derived at replay time (``whole``: one chunk; ``per_token``: one token per chunk). The plans
follow the corpus plan (section 4 of ``corpus.md``): fixed sizes, every two-way split for short
outputs, and seeded random lengths. They are a function of the output length alone, so a re-record
with the same output gives the same plans.
"""

from __future__ import annotations

import random

FIXED_SIZES = (1, 2, 3, 5, 7, 11, 23)
RANDOM_PLANS = 30
RANDOM_LONGEST_CHUNK = 8
TWO_WAY_SPLIT_LIMIT = 32


def chunk_plans(count: int) -> dict[str, list[int] | None]:
    if count < 1:
        raise ValueError("an output needs at least one token to be chunked")
    plans: dict[str, list[int] | None] = {"whole": None, "per_token": None}
    for size in FIXED_SIZES:
        if 1 < size < count:
            plans[f"size-{size}"] = fixed(count, size)
    if count <= TWO_WAY_SPLIT_LIMIT:
        for first in range(1, count):
            plans[f"split-{first}"] = [first, count - first]
    for seed in range(1, RANDOM_PLANS + 1):
        plans[f"random-{seed}"] = seeded(count, seed)
    return plans


def fixed(count: int, size: int) -> list[int]:
    lengths = [size] * (count // size)
    if count % size:
        lengths.append(count % size)
    return lengths


def seeded(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    lengths: list[int] = []
    left = count
    while left > 0:
        length = min(rng.randint(1, RANDOM_LONGEST_CHUNK), left)
        lengths.append(length)
        left -= length
    return lengths

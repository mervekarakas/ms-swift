# pair_sampler.py
import random
from collections import defaultdict
from typing import Dict, Iterator, List
from torch.utils.data import Sampler


class PairRepeatSampler(Sampler[int]):
    """
    Pair-id based *index* sampler for pointwise BT + GRPO.

    Guarantees that, within each rank:
      - we sample pair_ids (not rows)
      - each pair_id contributes exactly:
            [idx_side0] * K  +  [idx_side1] * K
        where K = num_generations (mini_repeat_count)

    This makes each "prompt group" have K samples (as GRPO expects),
    and keeps both sides on the same rank (pair-level sharding).
    """

    def __init__(
        self,
        data_source,
        mini_repeat_count: int,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_source = data_source
        self.n = len(data_source)

        self.mini_repeat_count = int(mini_repeat_count)  # K = num_generations
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)

        # Epoch used for reshuffling; if set_epoch is never called, stays 0
        self.epoch = 0

        # Build mapping once: pair_id -> {side: idx}
        pair_to: Dict[int, Dict[int, int]] = defaultdict(dict)
        for idx in range(self.n):
            ex = data_source[idx]
            pid = int(ex["pair_id"])
            side = int(ex["side"])
            pair_to[pid][side] = idx

        # Keep only complete pairs (must have both side=0 and side=1)
        self.pair_to = {pid: d for pid, d in pair_to.items() if 0 in d and 1 in d}
        if len(self.pair_to) == 0:
            raise ValueError("No complete pairs found (need both side=0 and side=1 for each pair_id).")

        self.pairs: List[int] = list(self.pair_to.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        pairs = list(self.pairs)

        # shuffle at pair level
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(pairs)

        # shard at PAIR LEVEL: ensures both sides stay on same rank
        pairs = pairs[self.rank :: self.world_size]

        # emit: side0 repeated K, then side1 repeated K
        for pid in pairs:
            idx0 = self.pair_to[pid][0]
            idx1 = self.pair_to[pid][1]

            for _ in range(self.mini_repeat_count):
                yield idx0
            for _ in range(self.mini_repeat_count):
                yield idx1

    def __len__(self) -> int:
        # approximate is fine; this is mainly for progress bars / sanity
        local_pairs = len(self.pairs[self.rank :: self.world_size])
        return local_pairs * 2 * self.mini_repeat_count

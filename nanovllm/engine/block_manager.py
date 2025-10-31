from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
        
        # Ensure num_cached_tokens never equals or exceeds sequence length
        if seq.num_cached_tokens >= len(seq):
            import logging
            logging.warning(
                "allocate: seq=%d has num_cached=%d >= len=%d, capping to len-1",
                seq.seq_id, seq.num_cached_tokens, len(seq)
            )
            seq.num_cached_tokens = max(len(seq) - 1, 0)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

    def rollback(self, seq: Sequence, target_len: int):
        assert target_len >= seq.num_cached_tokens, "Cannot rollback cached prefix"
        if target_len >= len(seq):
            return

        # Remove surplus tokens from the sequence buffer.
        del seq.token_ids[target_len:]
        seq.num_tokens = len(seq.token_ids)
        if seq.token_ids:
            seq.last_token = seq.token_ids[-1]

        # Release unused KV blocks.
        target_num_blocks = (target_len + self.block_size - 1) // self.block_size if target_len > 0 else 0
        while len(seq.block_table) > target_num_blocks:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        if not seq.block_table:
            return

        last_block_id = seq.block_table[-1]
        last_block = self.blocks[last_block_id]
        if target_len % self.block_size == 0:
            if last_block.hash == -1:
                token_ids = seq.block(len(seq.block_table) - 1)
                prefix = self.blocks[seq.block_table[-2]].hash if len(seq.block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
        else:
            # Partial block cannot be cached.
            last_block.hash = -1
            last_block.token_ids = []

    def max_speculative_tokens(self, seq: Sequence, draft_budget: int) -> int:
        if draft_budget <= 0:
            return 0

        available = 0
        free_blocks = len(self.free_block_ids)
        seq_len = len(seq)

        for _ in range(draft_budget):
            seq_len += 1
            if seq_len % self.block_size == 1:
                if free_blocks == 0:
                    break
                free_blocks -= 1
            available += 1

        return available

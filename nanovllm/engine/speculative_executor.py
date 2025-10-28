from typing import List, Tuple

import torch

from nanovllm.config import Config
from nanovllm.engine.runner import CpuDraftRunner, ModelRunner, RunnerOutput
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


class SpeculativeExecutor:

    def __init__(
        self,
        gpu_model: ModelRunner,
        cpu_model: CpuDraftRunner,
        scheduler: Scheduler,
        config: Config,
    ) -> None:
        self.gpu_model = gpu_model
        self.cpu_model = cpu_model
        self.scheduler = scheduler
        self.failure_times = 0
        self.speculative_max_retries = config.speculative_max_retries
        self.draft_budget = config.draft_num_tokens

    def can_step(self) -> bool:
        return self.failure_times <= self.speculative_max_retries

    def step(self, seqs: List[Sequence]) -> Tuple[List[int], int, bool]:
        if not seqs:
            return [], 0, True

        cpu_output: RunnerOutput = self.cpu_model.run(seqs, False)
        assert cpu_output.token_ids is not None
        draft_batches = cpu_output.token_ids
        assert len(draft_batches) == len(seqs)
        draft_prob_batches = cpu_output.probs or [torch.empty((0, 0)) for _ in seqs]
        if len(draft_prob_batches) < len(seqs):
            draft_prob_batches = list(draft_prob_batches) + [torch.empty((0, 0)) for _ in range(len(seqs) - len(draft_prob_batches))]

        block_manager = self.scheduler.block_manager

        sequence_snapshots = []
        for seq in seqs:
            sequence_snapshots.append({
                "token_ids": list(seq.token_ids),
                "num_tokens": seq.num_tokens,
                "last_token": seq.last_token,
                "num_cached_tokens": seq.num_cached_tokens,
                "block_table": list(seq.block_table),
                "status": seq.status,
            })

        block_snapshot = dict(
            free_block_ids=list(block_manager.free_block_ids),
            used_block_ids=set(block_manager.used_block_ids),
            hash_to_block_id=dict(block_manager.hash_to_block_id),
            blocks=[(blk.ref_count, blk.hash, list(blk.token_ids)) for blk in block_manager.blocks],
        )

        appended_counts = [0] * len(seqs)

        for idx, seq in enumerate(seqs):
            tokens = draft_batches[idx]
            if not tokens:
                continue
            for token in tokens:
                seq.append_token(token)
                appended_counts[idx] += 1
                block_manager.may_append(seq)
            seq.num_cached_tokens = sequence_snapshots[idx]["num_cached_tokens"]

        _, logits = self.gpu_model.run_with_logits(seqs, True)
        if logits is None:
            logits = torch.empty((0, 0))

        accepted_counts = [0] * len(seqs)
        fallback_tokens: List[int | None] = [None] * len(seqs)

        offset = 0
        with torch.no_grad():
            for idx, seq in enumerate(seqs):
                count = appended_counts[idx]
                tokens = draft_batches[idx]
                draft_probs = draft_prob_batches[idx] if idx < len(draft_prob_batches) else None
                if count == 0:
                    continue
                seq_logits = logits[offset:offset + count]
                offset += seq_logits.size(0)
                temperature = max(seq.temperature, 1e-6)
                for step_idx, token in enumerate(tokens):
                    if step_idx >= seq_logits.size(0):
                        break
                    gpu_logits = seq_logits[step_idx] / temperature
                    gpu_probs = torch.softmax(gpu_logits, dim=-1)

                    if (
                        draft_probs is None
                        or draft_probs.numel() == 0
                        or step_idx >= draft_probs.size(0)
                    ):
                        draft_probs_step = torch.zeros_like(gpu_probs)
                    else:
                        draft_probs_step = draft_probs[step_idx]
                        if draft_probs_step.device != gpu_probs.device:
                            draft_probs_step = draft_probs_step.to(gpu_probs.device)
                        if draft_probs_step.dtype != gpu_probs.dtype:
                            draft_probs_step = draft_probs_step.to(gpu_probs.dtype)

                    token_prob_gpu_val = float(gpu_probs[token].item())
                    token_prob_draft_val = float(draft_probs_step[token].item()) if draft_probs_step.numel() else 0.0

                    accept_token = False
                    if token_prob_gpu_val >= token_prob_draft_val or token_prob_draft_val <= 0.0:
                        accept_token = True
                    else:
                        ratio = token_prob_gpu_val / token_prob_draft_val
                        ratio = min(ratio, 1.0)
                        u = torch.rand(1, device=gpu_probs.device).item()
                        if u <= ratio:
                            accept_token = True

                    if accept_token:
                        accepted_counts[idx] += 1
                        continue

                    diff_probs = (gpu_probs - draft_probs_step).clamp_min(0)
                    diff_sum = diff_probs.sum()
                    if diff_sum <= 0:
                        fallback = int(torch.multinomial(gpu_probs, 1).item())
                    else:
                        fallback = int(torch.multinomial(diff_probs / diff_sum, 1).item())
                    fallback_tokens[idx] = fallback
                    break

        for seq, snapshot in zip(seqs, sequence_snapshots):
            seq.token_ids = list(snapshot["token_ids"])
            seq.num_tokens = snapshot["num_tokens"]
            seq.last_token = snapshot["last_token"]
            seq.block_table = list(snapshot["block_table"])
            seq.num_cached_tokens = snapshot["num_cached_tokens"]
            seq.status = snapshot["status"]

        block_manager.free_block_ids = block_manager.free_block_ids.__class__(block_snapshot["free_block_ids"])
        block_manager.used_block_ids = set(block_snapshot["used_block_ids"])
        block_manager.hash_to_block_id = dict(block_snapshot["hash_to_block_id"])
        for blk, (ref_count, blk_hash, token_ids) in zip(block_manager.blocks, block_snapshot["blocks"]):
            blk.ref_count = ref_count
            blk.hash = blk_hash
            blk.token_ids = list(token_ids)

        commit_records: list[tuple[Sequence, int]] = []
        last_gpu_tokens: List[int] = [snapshot["last_token"] for snapshot in sequence_snapshots]
        any_mismatch = False

        for idx, seq in enumerate(seqs):
            tokens = draft_batches[idx]
            accepted = accepted_counts[idx]
            fallback = fallback_tokens[idx]

            for step_idx in range(accepted):
                token = tokens[step_idx]
                commit_records.append((seq, token))
                last_gpu_tokens[idx] = token

            if accepted < len(tokens):
                any_mismatch = True
                if fallback is not None:
                    commit_records.append((seq, fallback))
                    last_gpu_tokens[idx] = fallback

        if commit_records:
            seq_batch, token_batch = zip(*commit_records)
            seq_batch = list(seq_batch)
            token_batch = list(token_batch)
            for seq, token in zip(seq_batch, token_batch):
                self.scheduler.postprocess([seq], [token])
                seq.num_cached_tokens = max(seq.num_tokens-1,0)
                if not seq.is_finished:
                    self.scheduler.block_manager.may_append(seq)

        committed = len(commit_records)

        if any_mismatch:
            self.failure_times += 1
        else:
            self.failure_times = 0

        return last_gpu_tokens, committed, True

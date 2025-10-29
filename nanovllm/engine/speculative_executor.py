from typing import List, Tuple

import torch

from nanovllm.config import Config
from nanovllm.engine.runner import CpuDraftRunner, ModelRunner, RunnerOutput
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


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
        self.block_manager = self.scheduler.block_manager

    def can_step(self) -> bool:
        return self.failure_times <= self.speculative_max_retries

    def step(self, seqs: List[Sequence]) -> Tuple[List[int], int, bool]:
        if not seqs:
            return [], 0, True

        cpu_output: RunnerOutput = self.cpu_model.run(seqs, False)
        assert cpu_output.token_ids is not None
        draft_batches = list(cpu_output.token_ids)
        assert len(draft_batches) == len(seqs)

        if cpu_output.probs is None:
            draft_prob_batches = [torch.empty((0, 0)) for _ in seqs]
        else:
            draft_prob_batches = list(cpu_output.probs)
            if len(draft_prob_batches) < len(seqs):
                draft_prob_batches.extend(torch.empty((0, 0)) for _ in range(len(seqs) - len(draft_prob_batches)))

        truncated_tokens: list[list[int]] = []
        truncated_probs: list[torch.Tensor] = []
        total_draft = 0

        for seq, tokens, probs in zip(seqs, draft_batches, draft_prob_batches):
            max_tokens = self.block_manager.max_speculative_tokens(seq, self.draft_budget)
            available = min(len(tokens), max_tokens)
            truncated_tokens.append(tokens[:available])
            if available > 0 and probs.dim() >= 2 and probs.size(0) >= available:
                truncated_probs.append(probs[:available])
            else:
                truncated_probs.append(probs.new_empty((0, 0)))
            total_draft += available

        if total_draft == 0:
            self.failure_times += 1
            token_ids = self.gpu_model.run(seqs, False)
            return token_ids, 0, False

        base_lengths = [len(seq) for seq in seqs]
        base_cached = [seq.num_cached_tokens for seq in seqs]

        for idx, seq in enumerate(seqs):
            for token in truncated_tokens[idx]:
                if seq.block_table:
                    self.block_manager.may_append(seq)
                seq.append_token(token)
            seq.num_cached_tokens = base_cached[idx]

        with torch.inference_mode():
            _, logits = self.gpu_model.run_with_logits(seqs, is_prefill=True)

        # for idx, seq in enumerate(seqs):
        #     self.block_manager.rollback(seq, base_lengths[idx])
        #     seq.num_cached_tokens = base_cached[idx]

        offset = 0
        any_reject = False

        for idx, seq in enumerate(seqs):
            draft_tokens = truncated_tokens[idx]
            draft_probs = truncated_probs[idx]
            temperature = max(seq.temperature, 1e-6)
            pref_len = len(draft_tokens) + 1
            seq_logits = logits[offset:offset + pref_len]
            offset += pref_len

            verify_logits = seq_logits[:-1] if draft_tokens else seq_logits.new_empty((0, seq_logits.size(-1)))
            next_logits = seq_logits[-1]

            accepted_steps = 0
            rejected = False
            terminated = False
            base_completion = base_lengths[idx] - seq.num_prompt_tokens

            for step_idx, token in enumerate(draft_tokens):
                logits_step = verify_logits[step_idx] / temperature
                p = torch.softmax(logits_step, dim=-1)
                if draft_probs.numel() == 0:
                    q = torch.zeros_like(p).fill_(1.0 / p.size(-1))
                else:
                    q = draft_probs[step_idx].to(p.device, dtype=p.dtype).clamp_min_(1e-9)
                token_prob_p = p[token].clamp_min(1e-9)
                token_prob_q = q[token].clamp_min(1e-9)
                accept_prob = torch.clamp(token_prob_p / token_prob_q, max=1.0)
                if torch.rand((), device=p.device) <= accept_prob:
                    accepted_steps += 1
                    completion_tokens = base_completion + accepted_steps
                    if (not seq.ignore_eos and token == self.scheduler.eos) or completion_tokens >= seq.max_tokens:
                        keep_len = base_lengths[idx] + accepted_steps
                        self.block_manager.rollback(seq, keep_len)
                        terminated = True
                        break
                    continue

                residual = (p - q).clamp_min_(0.0)
                residual_sum = residual.sum()
                if residual_sum <= 0:
                    residual = p
                    residual_sum = residual.sum()
                residual = residual / residual_sum
                new_token = int(torch.multinomial(residual, 1).item())

                target_len = base_lengths[idx] + accepted_steps
                self.block_manager.rollback(seq, target_len)
                if seq.block_table:
                    self.block_manager.may_append(seq)
                seq.append_token(new_token)
                accepted_steps += 1
                rejected = True
                any_reject = True
                break

            if not rejected and not terminated:
                temperature_tensor = torch.tensor([temperature], dtype=next_logits.dtype, device=next_logits.device)
                next_token = int(self.gpu_model.sampler(next_logits.unsqueeze(0), temperature_tensor).item())
                if seq.block_table:
                    self.block_manager.may_append(seq)
                seq.append_token(next_token)

        committed = 0
        for idx, seq in enumerate(seqs):
            start = base_lengths[idx]
            new_tokens_count = max(len(seq) - start, 0)
            finished = False
            base_completion = base_lengths[idx] - seq.num_prompt_tokens
            for rel_idx in range(new_tokens_count):
                token = seq.token_ids[start + rel_idx]
                completion_tokens = base_completion + rel_idx + 1
                if (not seq.ignore_eos and token == self.scheduler.eos) or completion_tokens >= seq.max_tokens:
                    finish_len = start + rel_idx + 1
                    del seq.token_ids[finish_len:]
                    seq.num_tokens = finish_len
                    seq.last_token = seq.token_ids[-1] if seq.token_ids else seq.last_token
                    new_tokens_count = finish_len - start
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.scheduler.running:
                        self.scheduler.running.remove(seq)
                    finished = True
                    break
            committed += new_tokens_count
            if not finished:
                seq.num_cached_tokens = max(len(seq) - 1, 0)

        if any_reject:
            self.failure_times += 1
        else:
            self.failure_times = 0

        return [], committed, True

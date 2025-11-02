"""
Minimal fix for CPU draft runner - avoid expensive save_state/load_state operations
This is a drop-in replacement that should immediately improve performance
"""

import torch
from llama_cpp import Llama

from nanovllm.config import Config
from nanovllm.engine.runner.base import RunnerBase, RunnerOutput
from nanovllm.engine.sequence import Sequence


class CpuDraftRunner(RunnerBase):
    """
    Minimal optimization: Avoid save_state/load_state by tracking sequences manually
    Key changes:
    1. Don't save/load full KV state - too expensive!
    2. Track sequences with simple token lists
    3. Re-evaluate from start when context changes (still faster than state save/load)
    """

    def __init__(self, config: Config):
        assert config.draft_model is not None and config.draft_model.endswith(".gguf")
        self.draft_num_tokens = config.draft_num_tokens
        self.temperature_floor = 1e-6

        # Initialize llama.cpp
        self.llm = Llama(
            model_path=config.draft_model,
            n_ctx=config.llama_cpp_n_ctx,
            n_threads=config.llama_cpp_threads,
            n_gpu_layers=config.llama_cpp_n_gpu_layers,
            logits_all=True,
            verbose=False,
            use_mmap=True,  # Memory-mapped for faster loading
        )

        # Track sequences WITHOUT saving full KV state
        # Just track what tokens we've processed
        self._seq_cache: dict[int, list[int]] = {}
        self._last_seq_id = None

    def run(
        self,
        sequences: list[Sequence],
        is_prefill: bool,
    ) -> RunnerOutput:
        if is_prefill:
            return RunnerOutput()

        assert sequences, "CpuDraftRunner expects at least one sequence"

        batch_tokens: list[list[int]] = []
        batch_probs: list[torch.Tensor] = []

        for seq in sequences:
            seq_id = seq.seq_id
            current_tokens = seq.token_ids
            
            # Check if we can reuse context
            cached_tokens = self._seq_cache.get(seq_id, [])
            
            # Find common prefix length
            common_len = 0
            for i in range(min(len(cached_tokens), len(current_tokens))):
                if cached_tokens[i] == current_tokens[i]:
                    common_len = i + 1
                else:
                    break
            
            # Strategy: If we need to backtrack or switch context significantly,
            # just re-evaluate from scratch (faster than save/load state!)
            if seq_id != self._last_seq_id or common_len < len(cached_tokens):
                # Reset and rebuild context
                self.llm.reset()
                if current_tokens:
                    self.llm.eval(current_tokens)
                self._seq_cache[seq_id] = current_tokens.copy()
            elif common_len < len(current_tokens):
                # We can continue from where we left off
                new_tokens = current_tokens[common_len:]
                if new_tokens:
                    self.llm.eval(new_tokens)
                self._seq_cache[seq_id] = current_tokens.copy()
            
            self._last_seq_id = seq_id

            # Generate draft tokens
            new_tokens: list[int] = []
            step_probs: list[torch.Tensor] = []
            temperature = max(seq.temperature, self.temperature_floor)

            for _ in range(self.draft_num_tokens):
                logits = self.llm.eval_logits
                if logits is None or len(logits) == 0:
                    break
                
                # Fast tensor operations
                logits_t = torch.tensor(logits[-1], dtype=torch.float32)
                probs = torch.softmax(logits_t / temperature, dim=-1)
                step_probs.append(probs)
                
                next_token = int(torch.multinomial(probs, 1).item())
                new_tokens.append(next_token)
                
                # Advance context
                self.llm.eval([next_token])

            batch_tokens.append(new_tokens)
            if step_probs:
                probs_tensor = torch.stack(step_probs, dim=0)
            else:
                probs_tensor = torch.empty((0, 0))
            batch_probs.append(probs_tensor)

        return RunnerOutput(token_ids=batch_tokens, probs=batch_probs)


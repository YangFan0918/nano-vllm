import torch
from llama_cpp import Llama

from nanovllm.config import Config
from nanovllm.engine.runner.base import RunnerBase, RunnerOutput
from nanovllm.engine.sequence import Sequence


class CpuDraftRunner(RunnerBase):
    """
    llama.cpp-based draft runner (replaces HuggingFace). It keeps the same class name
    to avoid touching other modules. For each sequence we maintain a llama.cpp
    KV state to support incremental decoding across steps.
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
            verbose=False
        )
        # Base empty state
        self.llm.reset()
        self._base_state = self.llm.save_state()

        # Per-sequence KV state cache: seq_id -> (state_bytes, cached_len)
        self._states: dict[int, tuple[bytes, int]] = {}

    def _load_seq_state(self, seq_id: int) -> int:
        state = self._states.get(seq_id)
        if state is None:
            # Load empty base state
            self.llm.reset()
            if self._base_state:
                self.llm.load_state(self._base_state)
            return 0
        else:
            state_bytes, cached_len = state
            self.llm.load_state(state_bytes)
            return cached_len

    def _save_seq_state(self, seq_id: int, cur_len: int):
        state_bytes = self.llm.save_state()
        self._states[seq_id] = (state_bytes, cur_len)

    def _eval_to_len(self, input_ids: list[int], cur_len: int) -> int:
        if cur_len < len(input_ids):
            pending = input_ids[cur_len:]
            if pending:
                self.llm.eval(pending)
            return len(input_ids)
        return cur_len

    def run(
        self,
        sequences: list[Sequence],
        is_prefill: bool,
    ) -> RunnerOutput:
        if is_prefill:
            return RunnerOutput()

        assert sequences, "CpuDraftRunner (llama.cpp) expects at least one sequence"

        batch_tokens: list[list[int]] = []
        batch_probs: list[torch.Tensor] = []

        for seq in sequences:
            # Restore KV state and catch up to current committed prompt length
            cur_len = self._load_seq_state(seq.seq_id)
            cur_len = self._eval_to_len(seq.token_ids, cur_len)
            # Persist baseline at committed length only (speculative tokens are not saved)
            self._save_seq_state(seq.seq_id, len(seq.token_ids))

            new_tokens: list[int] = []
            step_probs: list[torch.Tensor] = []
            steps = self.draft_num_tokens
            temperature = max(seq.temperature, self.temperature_floor)

            for _ in range(steps):
                logits = self.llm.eval_logits
                if logits is None or len(logits) == 0:
                    break
                logits_t = torch.tensor(logits[-1], dtype=torch.float32)
                probs = torch.softmax(logits_t / temperature, dim=-1)
                step_probs.append(probs)
                next_token = int(torch.multinomial(probs, 1).item())
                new_tokens.append(next_token)
                # Advance context with sampled token
                self.llm.eval([next_token])

            # IMPORTANT: do NOT persist state including speculative tokens.
            # The saved state remains at committed_len to allow GPU-side rollback.

            batch_tokens.append(new_tokens)
            if step_probs:
                probs_tensor = torch.stack(step_probs, dim=0)
            else:
                probs_tensor = torch.empty((0, 0))
            batch_probs.append(probs_tensor)

        return RunnerOutput(token_ids=batch_tokens, probs=batch_probs)

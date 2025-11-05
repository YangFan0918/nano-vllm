import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    enable_speculative_sampling: bool = False
    draft_model: str | None = None
    draft_num_tokens: int = 4
    draft_workers: int = 1
    speculative_max_retries: int = 10
    # llama.cpp (CPU draft runner) settings
    llama_cpp_threads: int = 10  # fixed by default per user request
    llama_cpp_n_ctx: int = 0     # 0 -> use max_model_len
    llama_cpp_n_gpu_layers: int = 0
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        if self.enable_speculative_sampling:
            # Require GGUF file path for llama.cpp draft model
            assert self.draft_model is not None, "draft_model must be provided when speculative sampling is enabled"
            assert os.path.isfile(self.draft_model), "draft_model must be a GGUF file path"
            assert self.draft_model.endswith('.gguf'), "draft_model must be a .gguf file for llama.cpp"
            assert self.draft_num_tokens > 0
            assert self.draft_workers > 0
            assert self.speculative_max_retries >= 0
            # Optional env override for llama.cpp thread count
            env_llama_threads = os.getenv("NANOVLLM_LLAMA_CPP_THREADS")
            if env_llama_threads is not None:
                try:
                    self.llama_cpp_threads = int(env_llama_threads)
                except Exception:
                    pass
            if self.llama_cpp_threads <= 0:
                # Keep it simple: fixed 10 threads unless explicitly overridden.
                self.llama_cpp_threads = 10
            assert self.llama_cpp_threads > 0
            if self.llama_cpp_n_ctx <= 0:
                self.llama_cpp_n_ctx = self.max_model_len
            assert self.llama_cpp_n_ctx > 0
            assert self.llama_cpp_n_gpu_layers >= 0

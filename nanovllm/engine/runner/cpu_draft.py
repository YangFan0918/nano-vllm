import torch
from transformers import AutoModelForCausalLM

from nanovllm.config import Config
from nanovllm.engine.runner.base import RunnerBase, RunnerOutput
from nanovllm.engine.sequence import Sequence


class CpuDraftRunner(RunnerBase):
    def __init__(self, config: Config):
        assert config.draft_model is not None
        self.draft_num_tokens = config.draft_num_tokens
        self.device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.draft_model,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model.eval()

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
            input_ids = torch.tensor(
                seq.token_ids,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=self.draft_num_tokens,
                    do_sample=True,
                    temperature=seq.temperature,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            new_tokens = generated.sequences[0, input_ids.size(1):].tolist()
            batch_tokens.append(new_tokens)

            step_probs: list[torch.Tensor] = []
            for score in generated.scores[: len(new_tokens)]:
                logits = score[0]
                probs = torch.softmax(logits, dim=-1)
                step_probs.append(probs)

            if step_probs:
                probs_tensor = torch.stack(step_probs, dim=0).to(self.device)
            else:
                probs_tensor = torch.empty((0, self.model.config.vocab_size), device=self.device)
            batch_probs.append(probs_tensor)

        return RunnerOutput(token_ids=batch_tokens, probs=batch_probs)

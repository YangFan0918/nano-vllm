from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from nanovllm.engine.sequence import Sequence


@dataclass
class RunnerOutput:
    token_ids: list[list[int]] | None = None
    probs: list[torch.Tensor] | None = None


class RunnerBase(ABC):

    @abstractmethod
    def run(
        self,
        sequences: list[Sequence],
        is_prefill: bool,
    ) -> RunnerOutput:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

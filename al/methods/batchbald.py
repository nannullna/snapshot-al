# Reference: https://github.com/BlackHC/batchbald_redux

from typing import Iterable, Optional, List, Tuple, Any
from copy import deepcopy
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from toma import toma

from tqdm.auto import tqdm, trange

from .ensemble import EnsembleQuery
from ..pool import ActivePool
from ..query import QueryResult


def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N


class JointEntropy:
    """Random variables (all with the same # of categories $C$) can be added via `JointEntropy.add_variables`.
    `JointEntropy.compute` computes the joint entropy.
    `JointEntropy.compute_batch` computes the joint entropy of the added variables with each of the variables in the provided batch probabilities in turn."""

    def compute(self) -> torch.Tensor:
        """Computes the entropy of this joint entropy."""
        raise NotImplementedError()

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "JointEntropy":
        """Expands the joint entropy to include more terms."""
        raise NotImplementedError()

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None) -> torch.Tensor:
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        raise NotImplementedError()


class ExactJointEntropy(JointEntropy):
    joint_probs_M_K: torch.Tensor

    def __init__(self, joint_probs_M_K: torch.Tensor):
        self.joint_probs_M_K = joint_probs_M_K

    @staticmethod
    def empty(K: int, device=None, dtype=None) -> "ExactJointEntropy":
        return ExactJointEntropy(torch.ones((1, K), device=device, dtype=dtype))

    def compute(self) -> torch.Tensor:
        probs_M = torch.mean(self.joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(probs_M) * probs_M
        entropy = torch.sum(nats_M)
        return entropy

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "ExactJointEntropy":
        assert self.joint_probs_M_K.shape[1] == log_probs_N_K_C.shape[1]

        N, K, C = log_probs_N_K_C.shape
        joint_probs_K_M_1 = self.joint_probs_M_K.t()[:, :, None]

        probs_N_K_C = log_probs_N_K_C.exp()

        # Using lots of memory.
        for i in range(N):
            probs_i__K_1_C = probs_N_K_C[i][:, None, :].to(joint_probs_K_M_1, non_blocking=True)
            joint_probs_K_M_C = joint_probs_K_M_1 * probs_i__K_1_C
            joint_probs_K_M_1 = joint_probs_K_M_C.reshape((K, -1, 1))

        self.joint_probs_M_K = joint_probs_K_M_1.squeeze(2).t()
        return self

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None):
        assert self.joint_probs_M_K.shape[1] == log_probs_B_K_C.shape[1]

        B, K, C = log_probs_B_K_C.shape
        M = self.joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device)

        pbar = tqdm(total=B, desc="ExactJointEntropy.compute_batch", leave=False)

        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)
        def chunked_joint_entropy(chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int):
            chunked_probs_b_K_C = chunked_log_probs_b_K_C.exp()
            b = chunked_probs_b_K_C.shape[0]

            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.joint_probs_M_K.dtype,
                device=self.joint_probs_M_K.device,
            )
            for i in range(b):
                torch.matmul(
                    self.joint_probs_M_K,
                    chunked_probs_b_K_C[i].to(self.joint_probs_M_K, non_blocking=True),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K

            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C, dim=(1, 2)),
                non_blocking=True,
            )

            pbar.update(end - start)

        pbar.close()

        return output_entropies_B


def batch_multi_choices(probs_b_C, M: int):
    """
    probs_b_C: Ni... x C
    Returns:
        choices: Ni... x M
    """
    probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))

    # samples: Ni... x draw_per_xx
    choices = torch.multinomial(probs_B_C, num_samples=M, replacement=True)

    choices_b_M = choices.reshape(list(probs_b_C.shape[:-1]) + [M])
    return choices_b_M


def gather_expand(data, dim, index):
    if gather_expand.DEBUG_CHECKS:
        assert len(data.shape) == len(index.shape)
        assert all(dr == ir or 1 in (dr, ir) for dr, ir in zip(data.shape, index.shape))

    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]

    data = data.expand(new_data_shape)
    index = index.expand(new_index_shape)

    return torch.gather(data, dim, index)


gather_expand.DEBUG_CHECKS = False


class SampledJointEntropy(JointEntropy):
    """Random variables (all with the same # of categories $C$) can be added via `SampledJointEntropy.add_variables`.
    `SampledJointEntropy.compute` computes the joint entropy.
    `SampledJointEntropy.compute_batch` computes the joint entropy of the added variables 
    with each of the variables in the provided batch probabilities in turn."""

    sampled_joint_probs_M_K: torch.Tensor

    def __init__(self, sampled_joint_probs_M_K: torch.Tensor):
        self.sampled_joint_probs_M_K = sampled_joint_probs_M_K

    @staticmethod
    def empty(K: int, device=None, dtype=None) -> "SampledJointEntropy":
        return SampledJointEntropy(torch.ones((1, K), device=device, dtype=dtype))

    @staticmethod
    def sample(probs_N_K_C: torch.Tensor, M: int) -> "SampledJointEntropy":
        K = probs_N_K_C.shape[1]

        # S: num of samples per w
        S = M // K

        choices_N_K_S = batch_multi_choices(probs_N_K_C, S).long()

        expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]

        probs_N_K_K_S = gather_expand(expanded_probs_N_K_1_C, dim=-1, index=expanded_choices_N_1_K_S)
        # exp sum log seems necessary to avoid 0s?
        probs_K_K_S = torch.exp(torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False))
        samples_K_M = probs_K_K_S.reshape((K, -1))

        samples_M_K = samples_K_M.t()
        return SampledJointEntropy(samples_M_K)

    def compute(self) -> torch.Tensor:
        sampled_joint_probs_M = torch.mean(self.sampled_joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(sampled_joint_probs_M)
        entropy = torch.mean(nats_M)
        return entropy

    def add_variables(self, log_probs_N_K_C: torch.Tensor, M2: int) -> "SampledJointEntropy":
        K = self.sampled_joint_probs_M_K.shape[1]
        assert K == log_probs_N_K_C.shape[1]

        sample_K_M1_1 = self.sampled_joint_probs_M_K.t()[:, :, None]

        new_sample_M2_K = self.sample(log_probs_N_K_C.exp(), M2).sampled_joint_probs_M_K
        new_sample_K_1_M2 = new_sample_M2_K.t()[:, None, :]

        merged_sample_K_M1_M2 = sample_K_M1_1 * new_sample_K_1_M2
        merged_sample_K_M = merged_sample_K_M1_M2.reshape((K, -1))

        self.sampled_joint_probs_M_K = merged_sample_K_M.t()

        return self

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None):
        assert self.sampled_joint_probs_M_K.shape[1] == log_probs_B_K_C.shape[1]

        B, K, C = log_probs_B_K_C.shape
        M = self.sampled_joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device)

        pbar = tqdm(total=B, desc="SampledJointEntropy.compute_batch", leave=False)

        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)
        def chunked_joint_entropy(chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int):
            b = chunked_log_probs_b_K_C.shape[0]

            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.sampled_joint_probs_M_K.dtype,
                device=self.sampled_joint_probs_M_K.device,
            )
            for i in range(b):
                torch.matmul(
                    self.sampled_joint_probs_M_K,
                    chunked_log_probs_b_K_C[i].to(self.sampled_joint_probs_M_K, non_blocking=True).exp(),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K

            q_1_M_1 = self.sampled_joint_probs_M_K.mean(dim=1, keepdim=True)[None]

            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C / q_1_M_1, dim=(1, 2)) / M,
                non_blocking=True,
            )

            pbar.update(end - start)

        pbar.close()

        return output_entropies_B


class DynamicJointEntropy(JointEntropy):
    inner: JointEntropy
    log_probs_max_N_K_C: torch.Tensor
    N: int
    M: int

    def __init__(self, M: int, max_N: int, K: int, C: int, dtype=None, device=None):
        self.M = M
        self.N = 0
        self.max_N = max_N

        self.inner = ExactJointEntropy.empty(K, dtype=dtype, device=device)
        self.log_probs_max_N_K_C = torch.empty((max_N, K, C), dtype=dtype, device=device)

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "DynamicJointEntropy":
        C = self.log_probs_max_N_K_C.shape[2]
        add_N = log_probs_N_K_C.shape[0]

        assert self.log_probs_max_N_K_C.shape[0] >= self.N + add_N
        assert self.log_probs_max_N_K_C.shape[2] == C

        self.log_probs_max_N_K_C[self.N : self.N + add_N] = log_probs_N_K_C
        self.N += add_N

        num_exact_samples = C ** self.N
        if num_exact_samples > self.M:
            self.inner = SampledJointEntropy.sample(self.log_probs_max_N_K_C[: self.N].exp(), self.M)
        else:
            self.inner.add_variables(log_probs_N_K_C)

        return self

    def compute(self) -> torch.Tensor:
        return self.inner.compute()

    def compute_batch(self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None) -> torch.Tensor:
        """Computes the joint entropy of the added variables together with the batch (one by one)."""
        return self.inner.compute_batch(log_probs_B_K_C, output_entropies_B)


class EnsembleBatchBALD(EnsembleQuery):

    def __init__(self, model: nn.Module, pool: ActivePool, size: int = 1, device: torch.device = None, **kwargs):
        super().__init__(model, pool, size, device, **kwargs)
        self.descending = True


    def get_batchbald_batch(
        self, log_probs_N_K_C: torch.Tensor, batch_size: int, num_samples: int, dtype=None, device=None
    ) -> Tuple[List, List]:
        N, K, C = log_probs_N_K_C.shape

        batch_size = min(batch_size, N)

        candidate_indices = []
        candidate_scores = []

        if batch_size == 0:
            return (candidate_scores, candidate_indices)

        conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

        batch_joint_entropy = DynamicJointEntropy(
            num_samples, batch_size - 1, K, C, dtype=dtype, device=device
        )

        # We always keep these on the CPU.
        scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

        for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
            if i > 0:
                latest_index = candidate_indices[-1]
                batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

            shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

            batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

            scores_N -= conditional_entropies_N + shared_conditinal_entropies
            scores_N[candidate_indices] = -float("inf")

            candidate_score, candidate_index = scores_N.max(dim=0)

            candidate_indices.append(candidate_index.item())
            candidate_scores.append(candidate_score.item())

        return (candidate_scores, candidate_indices)


    def query(self, checkpoints: List[str], size: int, dataloader: Optional[Iterable] = None, **kwargs) -> QueryResult:
        dataloader = dataloader or self.pool.get_unlabeled_dataloader()

        # temporarily save the model's state dict to restore it after querying
        temp_state_dict = deepcopy(self.model.state_dict())

        query_results = []
        for ckpt in tqdm(checkpoints, desc=self.__class__.__name__, unit="model"):
            self.model.load_state_dict(torch.load(ckpt)["state_dict"])
            query_results.append(self._query_impl(size, dataloader, **kwargs))
        all_log_probs = self.postprocess(query_results)
        N, K, C = all_log_probs.shape

        # restore the model
        self.model.load_state_dict(temp_state_dict)

        # start querying here
        scores, indices = self.get_batchbald_batch(all_log_probs, size, num_samples=10000, dtype=torch.float64, device=self.device)
        
        return QueryResult(scores=scores, indices=indices)


    def _query_impl(self, size: int, dataloader: Iterable, **kwargs) -> QueryResult:
        log_probs = []
        self.model.eval()
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                logits = self.classify(imgs) # [B, C]
                log_probs.append(F.log_softmax(logits, dim=1))
        log_probs = torch.cat(log_probs, dim=0) # [N, C]
        return log_probs.unsqueeze(1) # [N, 1, C]


    def postprocess(self, query_results: List[Any]) -> torch.Tensor:
        all_log_probs = torch.cat(query_results, dim=1)
        return all_log_probs
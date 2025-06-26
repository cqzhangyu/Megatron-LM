# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

from megatron.core.parallel_state import (
    get_dbep_rank,
    get_dbep_group,
    get_dbep_instance_for_layer,
    get_expert_data_parallel_rank,
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_expert_tensor_and_model_parallel_group,
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_rank,
)
from megatron.core.config import ENABLE_EXPERIMENTAL
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.tensor_parallel import (
    all_to_all,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.moe_utils import (
    ModelCommProcessGroups,
    get_capacity,
    maybe_move_tensor_to_cpu,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from .token_dispatcher import MoETokenDispatcher

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

try:
    from deep_ep import Buffer

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

import torch

_buffer_normal = None
_buffer_dbep = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int, is_dbep: bool = False):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer_normal
    global _buffer_dbep
    if is_dbep:
        _buffer = _buffer_dbep
    else:
        _buffer = _buffer_normal
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    if is_dbep:
        _buffer_dbep = _buffer
    else:
        _buffer_normal = _buffer
    return _buffer


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(ctx, 
                x, 
                token_indices, 
                token_probs, 
                group, 
                is_dbep, 
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                previous_event=None):
        """Forward pass of fused dispatch."""
        buffer = get_buffer(group, get_hidden_bytes(x), is_dbep)
        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=None,
            async_finish=True,
            allocate_on_comm_stream=False,
        )

        ctx.group = group
        ctx.handle = handle
        ctx.is_dbep = is_dbep
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, event)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle, grad_event
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output), ctx.is_dbep)
        handle = ctx.handle

        grad_x, grad_token_probs, event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, grad_token_probs, None, None, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, is_dbep, previous_event=None):
        """Forward pass of fused combine."""
        buffer = get_buffer(group, get_hidden_bytes(x), is_dbep)
        combined_x, _, event = buffer.combine(
            x, handle=handle, async_finish=True, previous_event=None, allocate_on_comm_stream=False
        )
        ctx.handle = handle
        ctx.group = group
        ctx.is_dbep = is_dbep

        return combined_x, event

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output), ctx.is_dbep)
        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, None, None, None

def _indices_to_multihot(num_local_experts: int, indices: torch.Tensor, probs: torch.Tensor):
    """
    Converts a tensor of indices to a multihot vector.

    Args:
        indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
        probs (torch.Tensor): [num_tokens, topk] token probabilities.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - routing_map: Multihot vector.
            - probs: Multihot probabilities.
    """
    batch_size = indices.shape[0]
    multihot_routing_map = torch.zeros(
        (batch_size, num_local_experts), dtype=torch.long, device=indices.device
    )

    multihot_probs = torch.zeros(
        (batch_size, num_local_experts), dtype=torch.float, device=indices.device
    )

    mask = indices != -1
    valid_indices = indices[mask]
    row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
        mask.sum(dim=1)
    )
    multihot_routing_map[row_indices, valid_indices] = 1
    multihot_probs[row_indices, valid_indices] = probs[mask]
    return multihot_routing_map.bool(), multihot_probs


def get_permuted_hidden_states_by_experts(hidden_states: torch.Tensor, dispatched_indices: torch.Tensor, dispatched_probs: torch.Tensor, tokens_per_expert: torch.Tensor, router_dtype: str, permute_fusion: bool = False) -> torch.Tensor:
    
    num_local_experts = tokens_per_expert.shape[0]
    if ENABLE_EXPERIMENTAL and permute_fusion:
        dispatched_routing_map, dispatched_probs = fused_indices_to_multihot(
            dispatched_indices, dispatched_probs, num_local_experts
        )
    else:
        dispatched_routing_map, dispatched_probs = _indices_to_multihot(
            num_local_experts, dispatched_indices, dispatched_probs
        )
    hidden_shape_before_permute = hidden_states.shape
    assert dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
    hidden_states, permuted_probs, reversed_mapping_for_combine = permute(
        hidden_states,
        dispatched_routing_map,
        probs=dispatched_probs,
        num_out_tokens=sum(tokens_per_expert),
        fused=permute_fusion,
    )
    if router_dtype == "fp64":
        permuted_probs = permuted_probs.to(torch.float64)
    return hidden_states, permuted_probs, dispatched_routing_map, hidden_shape_before_permute, reversed_mapping_for_combine

class MoEDBEPDeepEPTokenDispatcher(MoETokenDispatcher):
    """
    Flex token dispatcher using DBEP and DeepEP.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ):
        """
        Initialize the Flex token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
            model_comm_pgs (ModelCommProcessGroups, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, model_comm_pgs=model_comm_pgs)

        self.num_experts = config.num_moe_experts
        self.router_topk = config.moe_router_topk
        self.router_dtype = config.moe_router_dtype
        self.capacity_factor = config.moe_expert_capacity_factor
        self.permute_fusion = config.moe_permute_fusion
        self.num_local_experts = num_local_experts
        self.dbep_ratio = config.dbep_ratio
        self.dbep_multiplier = config.dbep_multiplier
        self.dbep_size = self.ep_size * self.dbep_multiplier
        self.dbep_rank = get_dbep_rank()
        self.dbep_group = get_dbep_group()
        self.ep_group_id = self.dbep_rank // self.ep_size
        self.global_rank = torch.distributed.get_rank()
        assert self.tp_size * self.ep_size > 1, "Flex token dispatcher requires TPxEP > 1"
        assert (
            self.config.moe_enable_deepep
        ), "DeepEP is not enabled. Please set --moe-enable-deepep to use DeepEP backend."
        assert (
            self.config.moe_pad_expert_input_to_capacity is False
        ), "Flex token dispatcher does not support --moe-pad-expert-input-to-capacity"
        
        token_indices: Optional[torch.Tensor] = None
        token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None

        if not HAVE_DEEP_EP:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

    def set_shared_experts(self, shared_experts):
        raise NotImplementedError(
            "Shared expert overlap is not supported in Flex Token Dispatcher."
        )

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        num_local_tokens = routing_map.shape[0]

        # routing_map = routing_map.reshape(num_local_tokens, self.num_experts)
        # probs = probs.reshape(num_local_tokens, self.num_experts)
        # Convert the format of routing map from multihot to indices.
        token_probs, token_indices = torch.topk(probs, self.router_topk, dim=-1)
        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = token_probs == 0
            token_indices = token_indices.masked_fill(mask, -1)

        # DeepEP only supports float32 probs
        if token_probs.dtype != torch.float32:
            if token_probs.dtype in [torch.bfloat16, torch.float16]:
                print("DeepEP only supports float32 probs, please set --moe-router-dtype=fp32")
            token_probs = token_probs.float()  # downcast or upcast

        self.dbep_instance = get_dbep_instance_for_layer(self.layer_number)
        # expert_replicas = self.dbep_instance.get_expert_replicas()
        # expert_num_replicas = self.dbep_instance.get_expert_num_replicas()
        
        if self.dbep_ratio != 1.0:
            num_local_tokens_normal = int(num_local_tokens * (1-self.dbep_ratio))
            hidden_states_normal = hidden_states[:num_local_tokens_normal, :]
            hidden_states_dbep = hidden_states[num_local_tokens_normal:, :]

            token_probs_normal = token_probs[:num_local_tokens_normal, :]
            token_probs_dbep = token_probs[num_local_tokens_normal:, :]
            token_indices_normal = token_indices[:num_local_tokens_normal, :]
            token_indices_normal = token_indices_normal.to(device=torch.device('cpu'))
            token_indices_dbep = token_indices[num_local_tokens_normal:, :]
            token_indices_dbep = token_indices_dbep.to(device=torch.device('cpu'), non_blocking=True)

            num_tokens_per_expert_normal = routing_map[:num_local_tokens_normal, :].sum(dim=0).to(
                dtype=torch.int, device=torch.device('cpu')
            )
            num_tokens_per_expert_dbep = routing_map[num_local_tokens_normal:, :].sum(dim=0, dtype=torch.int)

            buffer_normal = get_buffer(self.dbep_group, get_hidden_bytes(hidden_states_normal), is_dbep=False)
            # compute num_tokens_per_replica_normal
            num_tokens_per_replica_normal = torch.zeros(
                self.dbep_size * self.num_local_experts, dtype=torch.int, device=torch.device('cpu')
            )

            self.dbep_instance.split_tokens_to_replicas(num_tokens_per_expert_normal, num_tokens_per_replica_normal)
            # print(f"rank {self.global_rank} num_tokens_per_replica_normal: {num_tokens_per_replica_normal}")
            
            # convert token_indices_normal from expert to replica indices
            self.dbep_instance.permute_indices(
                token_indices_normal,
                num_tokens_per_replica_normal
            )
            token_indices_normal = token_indices_normal.to(device=torch.device('cuda'))
            # print(f"rank {self.global_rank} token_indices_normal: {token_indices_normal}")

            # get_dispatch_layout for normal
            (
                num_tokens_per_rank_normal,
                num_tokens_per_rdma_rank_normal,
                num_tokens_per_expert_normal,
                is_token_in_rank_normal,
                previous_event,
            ) = buffer_normal.get_dispatch_layout(
                token_indices_normal,
                self.num_local_experts * self.dbep_size,
                previous_event=None,
                async_finish=False,
                allocate_on_comm_stream=False,
            )
            # print(f"rank {self.global_rank} num_tokens_per_expert_normal: {num_tokens_per_expert_normal}")

            # start normal dispatch
            hidden_states_normal, dispatched_indices_normal, dispatched_probs_normal, num_tokens_per_expert_normal, handle_normal, event_normal = (
                FusedDispatch.apply(
                    hidden_states_normal, 
                    token_indices_normal, 
                    token_probs_normal, 
                    self.dbep_group, 
                    False,
                    num_tokens_per_rank_normal,
                    num_tokens_per_rdma_rank_normal,
                    num_tokens_per_expert_normal,
                    is_token_in_rank_normal,
                )
            )
            self.handle_normal = handle_normal
            # event_normal.current_stream_wait()
            # print(f"rank {self.global_rank} num_tokens_per_expert_normal after dispatch: {num_tokens_per_expert_normal}")
            # print(f"rank {self.global_rank} dispatched_indices_normal: {dispatched_indices_normal}")
            # print(f"rank {self.global_rank} hidden_states_normal: {hidden_states_normal.shape}")
        else:
            hidden_states_dbep = hidden_states
            token_probs_dbep = token_probs
            # this detach is very important, otherwise the backward will fail
            token_indices_dbep = token_indices.detach().clone()

            num_tokens_per_expert_dbep = routing_map.sum(dim=0, dtype=torch.int)

        # prepare linear programming
        if self.dbep_ratio == 1.0 and self.dbep_size <= 8:
            self.dbep_instance.prepare_deepep(
                num_tokens_per_expert_dbep,
                token_indices_dbep
            )
        else:
            # sync dbep load
            num_global_tokens_per_expert_dbep = torch.empty(
                (self.dbep_size, self.num_experts), dtype=torch.int, device=torch.device("cuda")
            )
            allgather_event = torch.distributed.all_gather_into_tensor(
                num_global_tokens_per_expert_dbep,
                num_tokens_per_expert_dbep,
                group=self.dbep_group,
                async_op=True,
            )

            avg_tokens = num_local_tokens * self.router_topk
            if self.dbep_ratio != 1.0:
                # sync normal load
                num_global_tokens_per_rank_normal = num_tokens_per_rank_normal.clone()
                allreduce_event = torch.distributed.all_reduce(
                    num_global_tokens_per_rank_normal,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.dbep_group,
                    async_op=False,
                )
                gpu_load_normal = num_global_tokens_per_rank_normal.to(dtype=torch.float64, device=torch.device('cpu')) / avg_tokens
                # print(f"rank {self.global_rank} num_global_tokens_per_rank_normal: {num_global_tokens_per_rank_normal}")
            else:
                gpu_load_normal = torch.zeros(
                    (self.dbep_size), dtype=torch.float64, device=torch.device('cpu')
                )

            token_indices_dbep = token_indices_dbep.to(device=torch.device('cpu'))
            num_global_tokens_per_expert_dbep = num_global_tokens_per_expert_dbep.to(device=torch.device('cpu'))
            # print(f"rank {self.global_rank} num_global_tokens_per_expert_dbep: {num_global_tokens_per_expert_dbep}")

            # allreduce_event.wait()
            allgather_event.wait()
            expert_load_dbep = num_global_tokens_per_expert_dbep.sum(dim=0, dtype=torch.float64) / avg_tokens
            num_global_tokens_per_replica_dbep = torch.zeros(
                (self.dbep_size, self.dbep_size * self.num_local_experts), dtype=torch.int, device=torch.device('cpu')
            )
            # launch linear programming
            self.dbep_instance.get_optimal_load(expert_load_dbep, gpu_load_normal, num_global_tokens_per_expert_dbep, int(avg_tokens), num_global_tokens_per_replica_dbep)
            # print(f"rank {self.global_rank} num_global_tokens_per_replica_dbep: {num_global_tokens_per_replica_dbep}")

            num_token_per_replica_dbep = num_global_tokens_per_replica_dbep[self.dbep_rank, :].view(-1)
            # convert token_indices_dbep from expert to replica indices
            self.dbep_instance.permute_indices(
                token_indices_dbep,
                num_token_per_replica_dbep
            )
            token_indices_dbep = token_indices_dbep.to(device=torch.device('cuda'))
            # print(f"rank {self.global_rank} token_indices_dbep: {token_indices_dbep}")
        
        # get_dispatch_layout dbep
        buffer_dbep = get_buffer(self.dbep_group, get_hidden_bytes(hidden_states_dbep), is_dbep=True)

        # print(f"rank {self.global_rank} token_indices_dbep: {token_indices_dbep}")
        (
            num_tokens_per_rank_dbep,
            num_tokens_per_rdma_rank_dbep,
            num_tokens_per_expert_dbep,
            is_token_in_rank_dbep,
            previous_event,
        ) = buffer_dbep.get_dispatch_layout(
            token_indices_dbep,
            self.num_local_experts * self.dbep_size,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        # print(f"rank {self.global_rank} num_tokens_per_expert_dbep: {num_tokens_per_expert_dbep}")

        if self.dbep_ratio != 1.0:
            event_normal.current_stream_wait()

        # start DBEP dispatch
        hidden_states_dbep, dispatched_indices_dbep, dispatched_probs_dbep, num_tokens_per_expert_dbep, handle_dbep, event_dbep = (
            FusedDispatch.apply(
                hidden_states_dbep, 
                token_indices_dbep, 
                token_probs_dbep, 
                self.dbep_group, 
                False,
                num_tokens_per_rank_dbep,
                num_tokens_per_rdma_rank_dbep,
                num_tokens_per_expert_dbep,
                is_token_in_rank_dbep,
            )
        )
        self.handle_dbep = handle_dbep
        event_dbep.current_stream_wait()

        if self.dbep_ratio != 1.0:
            hidden_states = torch.cat(
                [hidden_states_normal, hidden_states_dbep], dim=0
            ).contiguous()
            dispatched_indices = torch.cat(
                [dispatched_indices_normal, dispatched_indices_dbep], dim=0
            ).contiguous()
            dispatched_probs = torch.cat(
                [dispatched_probs_normal, dispatched_probs_dbep], dim=0
            ).contiguous()
            num_tokens_per_expert = (
                num_tokens_per_expert_normal + num_tokens_per_expert_dbep
            ).contiguous()
            self.num_input_tokens_normal = hidden_states_normal.shape[0]
        else:
            hidden_states = hidden_states_dbep
            dispatched_indices = dispatched_indices_dbep
            dispatched_probs = dispatched_probs_dbep
            num_tokens_per_expert = num_tokens_per_expert_dbep
            self.num_input_tokens_normal = 0
        
        # print(f"rank {self.global_rank} num_tokens_per_expert: {num_tokens_per_expert}")
        # print(f"rank {self.global_rank} dispatched_indices: {dispatched_indices.view(-1).cpu()}")
        # print(f"rank {self.global_rank} dispatched hidden_states: {hidden_states[:, 0].view(-1).cpu()}")

        # permute
        (
            global_input_tokens, 
            permuted_probs, 
            self.dispatched_routing_map, 
            self.hidden_shape_before_permute, 
            self.reversed_mapping_for_combine, 
        ) = get_permuted_hidden_states_by_experts(
            hidden_states,
            dispatched_indices,
            dispatched_probs,
            num_tokens_per_expert,
            self.router_dtype,
            self.permute_fusion,
        )
        
        self.num_tokens_per_expert = num_tokens_per_expert

        return global_input_tokens, num_tokens_per_expert, permuted_probs

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert bias is None, "Bias is not supported in MoEFlexTokenDispatcher"
        # hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)
        # hidden_states = self._comm_manager.combine(hidden_states)

        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            fused=self.permute_fusion,
        )
        # print(f"rank {self.global_rank} hidden_states after unpermute: {hidden_states.shape}")

        if self.dbep_ratio != 1.0:
            hidden_states_normal = hidden_states[:self.num_input_tokens_normal, :]
            hidden_states_dbep = hidden_states[self.num_input_tokens_normal:, :]
            (
                hidden_states_normal,
                event_normal,
            ) = FusedCombine.apply(
                hidden_states_normal, 
                self.dbep_group, 
                self.handle_normal, 
                False
            )
            event_normal.current_stream_wait()
        else:
            hidden_states_dbep = hidden_states
        
        (
            hidden_states_dbep,
            event_dbep,
        ) = FusedCombine.apply(
            hidden_states_dbep, 
            self.dbep_group, 
            self.handle_dbep, 
            False
        )

        event_dbep.current_stream_wait()

        if self.dbep_ratio != 1.0:
            hidden_states = torch.cat(
                [hidden_states_normal, hidden_states_dbep], dim=0
            ).contiguous()
        else:
            hidden_states = hidden_states_dbep

        return hidden_states.view(self.hidden_shape), None

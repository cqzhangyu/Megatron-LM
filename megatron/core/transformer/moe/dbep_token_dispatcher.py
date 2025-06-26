# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import numpy
import math
import datetime

from megatron.core.parallel_state import (
    get_expert_model_parallel_rank,
    get_dbep_rank,
    get_dbep_group,
    get_dbep_instance_for_layer,
    get_data_parallel_group,
)
from megatron.core.tensor_parallel import (
    all_to_all,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.fused_a2a import fused_combine, fused_dispatch
from megatron.core.transformer.moe.moe_utils import (
    ModelCommProcessGroups,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from .token_dispatcher import MoETokenDispatcher

""" We use the following notation throughout this file:
     H: hidden size
     B: micro batch size
     S: sequence length
     TP: tensor model parallel size
     EP: expert model parallel size
     num_local_tokens: S/TP*B
     num_global_tokens: num_local_tokens*TP*EP
"""

class MoEDBEPTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll-based token dispatcher.

    The workflow of AlltoAll token dispatcher is as follows:
    (1) preprocess(): calculate necessary metadata for communication and permute
    (2) token_permutation(): permute->A2A(EP)->AG(TP)->sort_chunk(if num_local_experts>1)
    (3) token_unpermutation(): sort_chunk(if num_local_experts>1)->RS(TP)->A2A(EP)->unpermute
    """

    def __init__(
        self, 
        num_local_experts: int, 
        config: TransformerConfig,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(config=config, model_comm_pgs=model_comm_pgs)
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"

        assert not config.moe_pad_expert_input_to_capacity
        # assert self.config.moe_expert_capacity_factor is None

        self.num_dbep_experts = config.num_dbep_experts
        self.num_normal_experts = config.num_moe_experts - config.num_dbep_experts
        self.num_local_dbep_experts = self.num_dbep_experts // self.ep_size
        self.num_local_normal_experts = self.num_local_experts - self.num_local_dbep_experts
        self.dbep_multiplier = config.dbep_multiplier
        self.dbep_size = self.ep_size * self.dbep_multiplier
        self.dbep_rank = get_dbep_rank()
        self.dbep_group = get_dbep_group()
        self.ep_rank = get_expert_model_parallel_rank()
        self.global_rank = torch.distributed.get_rank()
        self.side_stream = torch.cuda.Stream()
        local_normal_expert_indices_offset = (
            self.ep_rank * self.num_local_normal_experts
        )
        self.local_normal_expert_indices = [
            local_normal_expert_indices_offset + i for i in range(self.num_local_normal_experts)
        ]

        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits_normal = None
        self.input_splits_dbep = None
        # [ep_size]. Represents the number of tokens received by the current rank from
        # other EP ranks.
        self.output_splits_normal = None
        self.output_splits_dbep = None
        # [tp_size]. Represents the number of tokens received by the current rank from
        # other TP ranks.
        self.output_splits_tp_normal = None
        self.permute_idx_device = torch.device("cuda") if self.config.moe_permute_fusion else None

        # used in permutation 2
        if self.num_normal_experts > 0:
            input_chunk_idxs_normal = torch.arange(
                self.num_normal_experts * self.tp_size, device=self.permute_idx_device
            )
            # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
            self.sort_input_by_local_experts_normal = input_chunk_idxs_normal.reshape(
                -1, self.num_local_normal_experts
            ).T.ravel()
            # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
            self.restore_output_by_local_experts_normal = input_chunk_idxs_normal.reshape(
                self.num_local_normal_experts, -1
            ).T.ravel()

        input_chunk_idxs_dbep = torch.arange(
            self.dbep_size * self.num_local_dbep_experts * self.tp_size, device=self.permute_idx_device
        )
        self.sort_input_by_local_experts_dbep = input_chunk_idxs_dbep.reshape(
            -1, self.num_local_dbep_experts
        ).T.ravel()
        self.restore_output_by_local_experts_dbep = input_chunk_idxs_dbep.reshape(
            self.num_local_dbep_experts, -1
        ).T.ravel()

        self.shared_experts = None
        self.expert_dist_log_path = config.expert_dist_log_path

    def token_permutation(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        This method performs the following steps:
        1. Preprocess the routing map to get metadata for communication and permutation.
        2. Permute input tokens for AlltoAll communication.
        3. Perform expert parallel AlltoAll communication.
        4. Sort tokens by local expert (if multiple local experts exist).

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        """
        Preprocess token routing map for AlltoAll communication and token permutation.

        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        # [num_experts], number of tokens assigned to each expert from the current rank's input.
        num_local_tokens_per_expert = routing_map.sum(dim=0, dtype=torch.int)

        self.dbep_instance = get_dbep_instance_for_layer(self.layer_number)
    
        # Dropless
        num_in_tokens = routing_map.size(0) * self.config.moe_router_topk

        # self.time_0 = torch.cuda.Event(enable_timing=True)
        # self.time_1 = torch.cuda.Event(enable_timing=True)
        # self.time_2 = torch.cuda.Event(enable_timing=True)
        # Gather normal metadata
        if self.num_local_normal_experts > 0:
            num_local_tokens_per_expert_normal = num_local_tokens_per_expert[:self.num_normal_experts]
            # ===================================================
            # Calculate input_splits, output_splits for alltoall/allgather in variable size.
            # ===================================================
            # [ep_size]. Represents the number of tokens sent by the current rank to other
            # EP ranks.
            self.input_splits_normal = (
                num_local_tokens_per_expert_normal.reshape(self.ep_size, self.num_local_normal_experts)
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )

            # Gather the global distribution of tokens across ranks.
            # num_global_tokens_per_expert represents the number of tokens sent to each
            # expert by all ranks.
            # [tp_size, ep_size, num_experts]
            num_global_tokens_per_expert_normal = (
                # CQ: this function is time consuming!
                gather_from_sequence_parallel_region(
                    num_local_tokens_per_expert_normal, group=self.tp_ep_group
                )
                .reshape(self.ep_size, self.tp_size, self.num_normal_experts)
                .transpose(0, 1)
            )
            # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]
            num_global_tokens_per_local_expert_normal = num_global_tokens_per_expert_normal[
                :, :, self.local_normal_expert_indices[0] : self.local_normal_expert_indices[-1] + 1
            ].contiguous()

            # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
            num_global_tokens_per_rank_normal = num_global_tokens_per_local_expert_normal.sum(axis=2)
            # [tp_size, ep_size] -> [ep_size]
            # self.output_splits represents the number of tokens received by the current rank
            # from other EP rank.
            # self.time_0.record()
            self.output_splits_normal = (
                num_global_tokens_per_rank_normal[self.tp_rank]
                .to(torch.device("cpu"))#, non_blocking=True)
                # this operation costs ~100us?
                .numpy()
            )
            # self.time_1.record()
            # print(f"rank {self.global_rank} output_splits_normal {self.output_splits_normal}")
            # [tp_size, ep_size] -> [tp_size]
            # self.output_splits_tp represents the number of tokens received by the current
            # rank from other TP rank.
            # self.output_splits_tp_normal = (
            #     num_global_tokens_per_rank_normal.sum(axis=1)
            #     .to(torch.device("cpu"), non_blocking=True)
            #     .numpy()
            # )
        
        
        # torch.cuda.current_stream().synchronize()
        # Gather DBEP metadata
        if self.num_local_normal_experts > 0:
            num_in_tokens_normal = self.input_splits_normal.sum()
            self.num_out_tokens_normal = self.output_splits_normal.sum()
            num_local_tokens_per_expert_dbep = num_local_tokens_per_expert[self.num_normal_experts:]
            # num_local_tokens_per_expert_dbep = torch.zeros([(self.num_dbep_experts + 1)], dtype=torch.int)
            # num_local_tokens_per_expert_dbep[:self.num_dbep_experts] = num_local_tokens_per_expert[self.num_normal_experts:]
            # num_local_tokens_per_expert_dbep[-1] = self.num_out_tokens_normal
            # num_local_tokens_per_expert_dbep = num_local_tokens_per_expert_dbep.cuda()
            num_local_tokens_per_expert_dbep = torch.cat([num_local_tokens_per_expert_dbep, torch.tensor([self.num_out_tokens_normal], dtype=torch.int, device=torch.cuda.current_device())], dim=0)
        else:
            num_in_tokens_normal = 0
            self.num_out_tokens_normal = 0
            # self.time_2.record()
            # print(f"rank {self.global_rank} num_in_tokens_normal {num_in_tokens_normal}")
            # append num_out_tokens_normal to the end of num_local_tokens_per_expert_dbep
        
            num_local_tokens_per_expert_dbep = num_local_tokens_per_expert

        # print(f"rank {self.global_rank} num_local_tokens_per_expert_dbep {num_local_tokens_per_expert_dbep}")
        
        # [dbep_size, num_dbep_experts(+1)]
        async_event_dbep = [None]
        # print(f"rank {self.global_rank} start DBEP all-gather {self.dbep_group}")
        num_global_tokens_per_expert_dbep = (
            gather_from_sequence_parallel_region(
                num_local_tokens_per_expert_dbep, group=self.dbep_group, async_op=True, async_event=async_event_dbep
            )
        )
        
        # if self.shared_experts is not None:
        #     self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        # Permutation 1: input to AlltoAll input
        self.hidden_shape_before_permute = hidden_states.shape

        # permute in side stream
        with torch.cuda.stream(self.side_stream):
            permutated_local_input_tokens, permuted_probs, self.reversed_local_input_permutation_mapping = permute(
                hidden_states,
                routing_map,
                probs=probs,
                num_out_tokens=num_in_tokens,
                fused=self.config.moe_permute_fusion,
                drop_and_pad=False,
            )

        if self.num_local_normal_experts > 1:
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert_normal = num_global_tokens_per_local_expert_normal.view(
                -1, self.num_local_normal_experts
            )
            if not self.config.moe_permute_fusion:
                self.num_global_tokens_per_local_expert_normal = (
                    self.num_global_tokens_per_local_expert_normal.to(
                        torch.device("cpu"), non_blocking=True
                    )
                )
        
        # print(f"rank {self.global_rank} permutated_local_input_tokens {permutated_local_input_tokens[:, 0]}")

        # normal all-to-all
        async_event_normal = [None]
        if self.num_local_normal_experts > 0:
            # [num_in_tokens_normal, hidden_size]
            permutated_local_input_tokens_normal = permutated_local_input_tokens[
                :num_in_tokens_normal , :
            ]
            permuted_probs_normal = permuted_probs[:num_in_tokens_normal]

            global_input_tokens_normal = all_to_all(
                self.ep_group, permutated_local_input_tokens_normal, self.output_splits_normal, self.input_splits_normal, async_op=True, async_event=async_event_normal
            )
            global_probs_normal = all_to_all(
                self.ep_group, permuted_probs_normal, self.output_splits_normal, self.input_splits_normal
            )
            # print(f"rank {self.global_rank} global_input_tokens_normal {global_input_tokens_normal[:, 0]}")

        # if self.shared_experts is not None:
        #     self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)

        # DBEP prepare
        if async_event_dbep[0] is not None:
            async_event_dbep[0].wait()
        
        num_global_tokens_per_expert_dbep = num_global_tokens_per_expert_dbep.reshape(self.dbep_size, -1).to(torch.device("cpu"))#, non_blocking=True)
        # print(f"rank {self.global_rank} num_global_tokens_per_expert_dbep {num_global_tokens_per_expert_dbep}")

        # [num_dbep_experts]
        expert_load_dbep = (
            num_global_tokens_per_expert_dbep[:, :self.num_dbep_experts].sum(axis=0)
            # convert to double
            .to(torch.float64) / num_in_tokens
        )

        # calculate the normal load for each gpu
        # [dbep_size]
        if self.num_local_normal_experts > 0:
            gpu_load_normal = num_global_tokens_per_expert_dbep[:, -1].contiguous().view(-1).to(torch.float64) / num_in_tokens
        else:
            gpu_load_normal = torch.zeros(self.dbep_size, dtype=torch.float64)

        # launch linear programming
        expert_replicas = self.dbep_instance.get_expert_replicas()
        replica_expert_offsets = self.dbep_instance.get_replica_expert_offsets()
        # print(f"rank {self.global_rank} expert_replicas {expert_replicas}")
        # print(f"rank {self.global_rank} replica_expert_offsets {replica_expert_offsets}")
        self.expert_replicas = torch.tensor(
            expert_replicas, dtype=torch.int, device=self.permute_idx_device
        )
        replica_expert_offsets = torch.tensor(
            replica_expert_offsets, dtype=torch.int, device=self.permute_idx_device
        )

        # [dbep_size, num_dbep_replicas]
        num_global_tokens_per_replica = torch.zeros(
            (self.dbep_size, self.dbep_size * self.num_local_dbep_experts), dtype=torch.int
        )

        self.dbep_instance.get_optimal_load(expert_load_dbep, gpu_load_normal, num_global_tokens_per_expert_dbep, int(num_in_tokens), num_global_tokens_per_replica)
        
        # [dbep_size * num_local_dbep_experts]
        self.num_local_tokens_per_replica = num_global_tokens_per_replica[self.dbep_rank, :].view(-1).to(device=self.permute_idx_device)
        # split the local tokens into chunks to experts' replicas
        # expert's replica offset -> # local tokens to replica
        # [dbep_size * num_local_dbep_experts]
        expert_replica_splits = self.num_local_tokens_per_replica[expert_replicas]

        self.side_stream.synchronize()
        if self.num_local_normal_experts > 0:
            permutated_local_input_tokens_dbep = permutated_local_input_tokens[
                num_in_tokens_normal: , :
            ]
            permuted_probs_dbep = permuted_probs[num_in_tokens_normal:]
        else:
            permutated_local_input_tokens_dbep = permutated_local_input_tokens
            permuted_probs_dbep = permuted_probs

        # Permutation 1 for DBEP
        permutated_local_input_tokens_dbep, permuted_probs_dbep = sort_chunks_by_idxs(permutated_local_input_tokens_dbep, expert_replica_splits, replica_expert_offsets, probs=permuted_probs_dbep, fused=self.config.moe_permute_fusion)

        # print(f"rank {self.global_rank} permutated_local_input_tokens_dbep {permutated_local_input_tokens_dbep[:, 0]}")
        
        # [dbep_size, num_local_dbep_experts]
        self.num_global_tokens_per_local_replica = (
            num_global_tokens_per_replica[:, self.dbep_rank * self.num_local_dbep_experts : (self.dbep_rank + 1) * self.num_local_dbep_experts]
            .contiguous()
        )
        # print(f"rank {self.global_rank} num_global_tokens_per_local_replica {self.num_global_tokens_per_local_replica}")

        # [dbep_size]
        self.output_splits_dbep = (
            self.num_global_tokens_per_local_replica
            .sum(axis=1)
            # .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        # print(f"rank {self.global_rank} output_splits_dbep {self.output_splits_dbep}")

        # [dbep_size]
        self.input_splits_dbep = (
            num_global_tokens_per_replica[self.dbep_rank, :]
            .reshape(self.dbep_size, self.num_local_dbep_experts)
            .sum(axis=1)
            .numpy()
        )
        # print(f"rank {self.global_rank} input_splits_dbep {self.input_splits_dbep}")
        
        # torch.cuda.current_stream().synchronize()

        # DBEP all-to-all
        async_event_dbep = [None]
        global_probs_dbep = all_to_all(
            self.dbep_group, permuted_probs_dbep, self.output_splits_dbep, self.input_splits_dbep, async_op=False
        )
        global_input_tokens_dbep = all_to_all(
            self.dbep_group, permutated_local_input_tokens_dbep, self.output_splits_dbep, self.input_splits_dbep, async_op=False, async_event=async_event_dbep
        )

        # wait for normal alltoall to complete
        if async_event_normal[0] is not None:
            async_event_normal[0].wait()

        # if self.tp_size > 1:
        #     if self.output_splits_tp is None:
        #         output_split_sizes = None
        #     else:
        #         output_split_sizes = self.output_splits_tp.tolist()
        #     global_input_tokens = gather_from_sequence_parallel_region(
        #         global_input_tokens, group=self.tp_group, output_split_sizes=output_split_sizes
        #     )

        # Permutation 2: Sort tokens by local expert.
        if self.num_local_normal_experts > 1:
            global_input_tokens_normal, global_probs_normal = sort_chunks_by_idxs(
                global_input_tokens_normal,
                self.num_global_tokens_per_local_expert_normal.ravel(),
                self.sort_input_by_local_experts_normal,
                probs=global_probs_normal,
                fused=self.config.moe_permute_fusion,
            )

            # print(f"rank {self.global_rank} global_input_tokens_normal {global_input_tokens_normal[:, 0]}")
        
        if async_event_dbep[0] is not None:
            async_event_dbep[0].wait()

        if self.config.moe_permute_fusion:
            split_sizes_dbep = self.num_global_tokens_per_local_replica.ravel().to(torch.device('cuda'))
        else:
            split_sizes_dbep = self.num_global_tokens_per_local_replica.ravel()
    
        global_input_tokens_dbep, global_probs_dbep = sort_chunks_by_idxs(
            global_input_tokens_dbep,
            split_sizes_dbep,
            self.sort_input_by_local_experts_dbep,
            probs=global_probs_dbep,
            fused=self.config.moe_permute_fusion,
        )
        # print(f"rank {self.global_rank} global_input_tokens_dbep {global_input_tokens_dbep[:, 0]}")

        if self.num_normal_experts > 0:
            global_input_tokens = torch.cat([global_input_tokens_normal, global_input_tokens_dbep], dim=0)
            global_probs = torch.cat([global_probs_normal, global_probs_dbep], dim=0)
        else:
            global_input_tokens = global_input_tokens_dbep.contiguous()
            global_probs = global_probs_dbep.contiguous()
        
        # if self.ep_rank == 0 and self.tp_rank == 0 and self.training:
        #     # Calculate the number of tokens for each global expert.
        #     # [num_experts]
        #     num_global_tokens_per_global_expert = num_global_tokens_per_expert.sum(axis=(0, 1)).to(
        #         torch.device("cpu"), non_blocking=True
        #     )
        #     # [num_experts]
        #     num_local_tokens_per_expert_cpu = num_local_tokens_per_expert.to(
        #         torch.device("cpu"), non_blocking=True
        #     )
        #     # [tp_size, ep_size, num_local_experts] -> [tp_size * ep_size * num_local_experts]
        #     # num_global_tokens_per_local_expert_cpu = num_global_tokens_per_local_expert.view(-1).contiguous().to(
        #     #     torch.device("cpu"), non_blocking=True
        #     # )

        #     torch.cuda.current_stream().synchronize()
        #     # Write expert distribution to a log file.
        #     with open(self.expert_dist_log_path, "a") as f:
        #         # [num_experts]
        #         f.write(f"Layer {self.layer_number} num_global_tokens_per_global_expert : {num_global_tokens_per_global_expert.tolist()}\n")
        #         # [num_experts]
        #         f.write(f"Layer {self.layer_number} num_local_tokens_per_expert : {num_local_tokens_per_expert_cpu.tolist()}\n")
        #         # [num_local_experts]
        #         # f.write(f"Layer {self.layer_number} num_global_tokens_per_local_expert : {num_global_tokens_per_local_expert.tolist()}\n")

        # [num_local_dbep_experts]
        num_tokens_per_local_replica = (
            self.num_global_tokens_per_local_replica.sum(axis=0)
        )

        if self.num_local_normal_experts > 0:
            # [tp_size, ep_size, num_local_experts] -> [num_local_experts]
            num_tokens_per_local_expert_normal = num_global_tokens_per_local_expert_normal.sum(dim=(0, 1))
            num_tokens_per_local_expert = torch.cat([num_tokens_per_local_expert_normal, num_tokens_per_local_replica], dim=0)
        else:
            num_tokens_per_local_expert = num_tokens_per_local_replica

        # if self.dbep_rank == 0:
        #     max_tokens_per_gpu = num_global_tokens_per_replica.reshape(self.dbep_size, self.dbep_size, self.num_local_dbep_experts).sum(axis=2).sum(axis=0).max().item()
        #     torch.cuda.current_stream().synchronize()
        #     print(f"max gpu load {max_tokens_per_gpu / num_in_tokens:.2f} for layer {self.layer_number}")
        # torch.cuda.current_stream().synchronize()
        # print(f"rank {self.global_rank} num_tokens_per_local_expert {num_tokens_per_local_expert}")
        # print(f"rank {self.global_rank} global_input_tokens.shape {global_input_tokens.shape}")
        # print(f"rank {self.global_rank} global_probs.shape {global_probs.shape}")

        return global_input_tokens, num_tokens_per_local_expert, global_probs

    def token_unpermutation(
        self, hidden_states: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        This method performs the following steps:
        1. Unsort tokens by local expert (if multiple local experts exist).
        2. Perform expert parallel AlltoAll communication to restore the original order.
        3. Unpermute tokens to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        hidden_states_normal = hidden_states[:self.num_out_tokens_normal, :]
        hidden_states_dbep = hidden_states[self.num_out_tokens_normal:, :]

        if self.config.moe_permute_fusion:
            split_sizes_dbep = self.num_global_tokens_per_local_replica.T.ravel().to(torch.device('cuda'))
        else:
            split_sizes_dbep = self.num_global_tokens_per_local_replica.T.ravel()

        # Unpermutation 2: DBEP
        hidden_states_dbep, _ = sort_chunks_by_idxs(
            hidden_states_dbep,
            split_sizes_dbep,
            self.restore_output_by_local_experts_dbep,
            fused=self.config.moe_permute_fusion,
        )
        
        # print(f"rank {self.global_rank} hidden_states_dbep {hidden_states_dbep[:, 0]}")

        # DBEP all-to-all
        async_event_dbep = [None]
        permutated_local_input_tokens_dbep = all_to_all(
            self.dbep_group, hidden_states_dbep, self.input_splits_dbep, self.output_splits_dbep, async_op=True, async_event=async_event_dbep
        )

        # Unpermute 2 normal
        if self.num_local_normal_experts > 1:
            hidden_states_normal, _ = sort_chunks_by_idxs(
                hidden_states_normal,
                self.num_global_tokens_per_local_expert_normal.T.ravel(),
                self.restore_output_by_local_experts_normal,
                fused=self.config.moe_permute_fusion,
            )
            # print(f"rank {self.global_rank} hidden_states_normal {hidden_states_normal[:, 0]}")

        # if self.tp_size > 1:
        #     if self.output_splits_tp is None:
        #         input_split_sizes = None
        #     else:
        #         input_split_sizes = self.output_splits_tp.tolist()
        #     hidden_states = reduce_scatter_to_sequence_parallel_region(
        #         hidden_states, group=self.tp_group, input_split_sizes=input_split_sizes
        #     )

        # normal all-to-all
        async_event_normal = [None]
        if self.num_local_normal_experts > 0:
            permutated_local_input_tokens_normal = all_to_all(
                self.ep_group, hidden_states_normal, self.input_splits_normal, self.output_splits_normal, async_op=True, async_event=async_event_normal
            )

        if async_event_dbep[0] is not None:
            async_event_dbep[0].wait()

        # DBEP unpermutation 1
        permutated_local_input_tokens_dbep, _ = sort_chunks_by_idxs(permutated_local_input_tokens_dbep, self.num_local_tokens_per_replica, self.expert_replicas, fused=self.config.moe_permute_fusion)

        if async_event_normal[0] is not None:
            async_event_normal[0].wait()

        if self.num_local_normal_experts > 0:
            permutated_local_input_tokens = torch.cat([permutated_local_input_tokens_normal, permutated_local_input_tokens_dbep], dim=0)
        else:
            permutated_local_input_tokens = permutated_local_input_tokens_dbep
        # print(f"rank {self.global_rank} permutated_local_input_tokens {permutated_local_input_tokens[:, 0]}")

        # if self.shared_experts is not None:
        #     self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
        #     self.shared_experts.post_forward_comm()

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=False,
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        # # Add shared experts output
        # if self.shared_experts is not None:
        #     shared_expert_output = self.shared_experts.get_output()
        #     output += shared_expert_output
        return output, None

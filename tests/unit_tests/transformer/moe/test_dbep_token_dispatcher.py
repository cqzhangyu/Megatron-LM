# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import copy
import dataclasses

import pytest
import torch
import random

import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent))

from megatron.core import parallel_state
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.cocktail import CocktailDataParallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.dbep_token_dispatcher import MoEDBEPTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class MoEModelTestContainer:
    def __init__(
        self,
        tp_size,
        ep_size,
        pp_size,
        dbep_multiplier,
        cp_size=1,
        moe_tp_size=None,
        data_parallel_random_init=False,
        num_moe_experts=8,
        num_dbep_experts=4,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_aux_loss_coeff=0.1,
        **kwargs,
    ):
        
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
            dbep_multiplier=dbep_multiplier,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=data_parallel_random_init)
        self.config = TransformerConfig(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
            pipeline_dtype=torch.bfloat16,
            params_dtype=torch.bfloat16,
            moe_router_topk=moe_router_topk,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_pad_expert_input_to_capacity=moe_pad_expert_input_to_capacity,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            num_layers=1,
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm", False),
            hidden_size=kwargs.get("hidden_size", 16),
            num_attention_heads=kwargs.get("num_attention_heads", 8),
            use_cpu_initialization=kwargs.get("use_cpu_initialization", True),
            sequence_parallel=tp_size > 1,
            add_bias_linear=False,
            moe_permute_fusion=kwargs.get("moe_permute_fusion", False),
            moe_enable_deepep=kwargs.get("moe_enable_deepep", False),
            dbep_multiplier=dbep_multiplier,
            num_dbep_experts=num_dbep_experts,
        )

        ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=True,
            use_distributed_optimizer=True,
            overlap_param_gather=True,
        )

        # init moe layer
        self.num_moe_experts = num_moe_experts
        self.num_dbep_experts = num_dbep_experts
        self.moe_layer = self.new_moe_layer()
        self.token_dispatcher = self.moe_layer.token_dispatcher

        for name, module in self.moe_layer.named_modules():
            print("-", name, module.__class__.__name__)

        model = CocktailDataParallel(config=self.config, ddp_config=ddp_config, module=self.moe_layer)

    def new_moe_layer(self, **kargs):
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.config.num_moe_experts, moe_grouped_gemm=self.config.moe_grouped_gemm
        )
        new_config = dataclasses.replace(self.config, **kargs)
        moe_layer = MoELayer(new_config, transformer_layer_spec.submodules.mlp.submodules).cuda()
        moe_layer.set_layer_number(1)
        return moe_layer

    def __del__(self):
        torch.distributed.barrier()
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def dispatcher_dropless_test(self):
        token_dispatcher = self.token_dispatcher
        bs = 1
        seql = 8
        # TODO: Find why setting manual seed can cause the test to fail
        # Manual seed to differentiate input data for each rank
        rank = torch.distributed.get_rank()
        torch.manual_seed(1000 + rank)
        random.seed(1000 + rank)
        hidden_states = torch.zeros((bs, seql, self.config.hidden_size), dtype=torch.float32)
        # Permute and then unpermute data are supposed to restore original data
        routing_map = torch.zeros((bs * seql, self.num_moe_experts), dtype=torch.bool)
        for i in range(seql):
            expert = random.randint(0, self.num_moe_experts - 1)
            hidden_states[0, i, :] = torch.ones(
                (self.config.hidden_size), dtype=torch.float32
            ) * (rank * 10 + expert)
            if self.num_moe_experts - self.num_dbep_experts > 0:
                routing_map[i, expert % (self.num_moe_experts - self.num_dbep_experts)] = True
            else:
                expert_2 = random.randint(1, self.num_moe_experts - 1)
                routing_map[i, (expert_2 + expert) % self.num_moe_experts] = True
            routing_map[i, expert % self.num_dbep_experts + self.num_moe_experts - self.num_dbep_experts] = True
        probs = torch.ones((bs * seql, self.num_moe_experts), dtype=torch.float32) / self.config.moe_router_topk
        hidden_states = hidden_states.cuda()
        ans = hidden_states
        hidden_states.requires_grad = True
        routing_map = routing_map.cuda()
        probs = probs.cuda()

        print("rank ", rank, "hidden_states", hidden_states[0, :, 0])

        (permuted_local_hidden_states, tokens_per_expert) = (
            token_dispatcher.token_permutation(hidden_states, probs, routing_map)
        )

        print("rank ", rank, "permuted_local_hidden_states", permuted_local_hidden_states[:, 0])

        restored_hidden_states, restored_bias = token_dispatcher.token_unpermutation(
            permuted_local_hidden_states
        )

        torch.cuda.current_stream().synchronize()
        print("rank ", rank, "restored_hidden_states", restored_hidden_states[0, :, 0])

        # reduce across TP rank equals to multiply data by a scale of ETP
        scale = self.config.expert_tensor_parallel_size
        restored_hidden_states = restored_hidden_states / scale

        assert torch.allclose(
            restored_hidden_states, ans
        ), "Restored hidden states do not match original hidden states"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, hidden_states)

        print("rank ", rank, "hidden_states.grad", hidden_states.grad[0, :, 0])
        print("rank ", rank, "ans", ans[0, :, 0])
        assert torch.allclose(
            hidden_states.grad, ans
        ), "Restored hidden states do not match original hidden states"


class TestDBEPTokenDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size, pp_size", [(1, 1, 2)])
    @pytest.mark.parametrize("dbep_multiplier", [2])
    @pytest.mark.parametrize("num_moe_experts, num_dbep_experts", [(8, 8)])
    @pytest.mark.parametrize("moe_router_topk", [2])
    @pytest.mark.parametrize("permute_fusion", [False])
    def test_forward_backward(self, tp_size, ep_size, pp_size, dbep_multiplier, num_moe_experts, num_dbep_experts, moe_router_topk, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            dbep_multiplier=dbep_multiplier,
            pp_size=pp_size,
            num_moe_experts=num_moe_experts,
            num_dbep_experts=num_dbep_experts,
            moe_router_topk=moe_router_topk,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=permute_fusion,
            moe_grouped_gemm=True,
        )

        container.dispatcher_dropless_test()

if __name__ == "__main__":
    # pytest.main([__file__])
    test = TestDBEPTokenDispatcher()
    test.setup_method(None)
    test.test_forward_backward(
        tp_size=1,
        ep_size=2,
        pp_size=1,
        dbep_multiplier=2,
        num_moe_experts=8,
        num_dbep_experts=8,
        moe_router_topk=2,
        permute_fusion=False,
    )
    test.teardown_method(None)

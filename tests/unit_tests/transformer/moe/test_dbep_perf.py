# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import copy
import dataclasses

import pytest
import torch
import random
import signal, traceback
import threading
import faulthandler

import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent))

from megatron.core import parallel_state
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.cocktail import CocktailDataParallel
from megatron.core.distributed import DistributedDataParallel
from megatron.core.models.gpt import GPTModel
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
        output_dir=None,
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
        # _set_random_seed(seed_=123, data_parallel_random_init=data_parallel_random_init)

        self.hidden_size = 1024
        self.num_attention_heads = 1
        self.sequence_length = 2048
        self.micro_batch_size = 8

        self.output_dir = output_dir

        self.config = TransformerConfig(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
            pipeline_dtype=torch.bfloat16,
            params_dtype=torch.bfloat16,
            moe_router_topk=moe_router_topk,
            moe_router_dtype="fp32",
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_pad_expert_input_to_capacity=moe_pad_expert_input_to_capacity,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            num_layers=2,
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm", False),
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
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

        self.config_base = copy.deepcopy(self.config)
        self.config_base.dbep_multiplier = None
        self.config_base.num_dbep_experts = None

        # init moe layer
        self.num_moe_experts = num_moe_experts
        self.num_dbep_experts = num_dbep_experts

        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.config.num_moe_experts, moe_grouped_gemm=self.config.moe_grouped_gemm
        )
        model = GPTModel(
            config=self.config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=32768,
            max_sequence_length=self.sequence_length,
            pre_process=True,
            post_process=True,
            fp16_lm_cross_entropy=False,
            parallel_output=True,
            share_embeddings_and_output_weights=True,
            position_embedding_type='rope',
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=False,
        )

        model_base = GPTModel(
            config=self.config_base,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=32768,
            max_sequence_length=self.sequence_length,
            pre_process=True,
            post_process=True,
            fp16_lm_cross_entropy=False,
            parallel_output=True,
            share_embeddings_and_output_weights=True,
            position_embedding_type='rope',
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=False,
        )

        if parallel_state.get_data_parallel_rank() == 0:
            print("Modules:")
            for name, module in model.named_modules():
                print("-", name, module.__class__.__name__)
    
        self.moe_layer = None
        for name, module in model.named_modules():
            if type(module) is MoELayer:
                if self.moe_layer is None:
                    self.moe_layer = module
                else:
                    self.moe_layer_2 = module
        
        self.moe_layers = torch.nn.Sequential(self.moe_layer, self.moe_layer_2)
        
        for name, module in model_base.named_modules():
            if type(module) is MoELayer:
                self.moe_layer_base = module
                break
        model.cuda(torch.cuda.current_device())
        model_base.cuda(torch.cuda.current_device())

        # self.model = CocktailDataParallel(config=self.config, ddp_config=ddp_config, module=model)
        self.token_dispatcher = self.moe_layer.token_dispatcher
        # self.model = CocktailDataParallel(config=self.config, ddp_config=ddp_config, module=self.moe_layer)
        self.model = CocktailDataParallel(config=self.config, ddp_config=ddp_config, module=self.moe_layers)
        self.token_dispatcher_base = self.moe_layer_base.token_dispatcher
        self.model_base = DistributedDataParallel(config=self.config_base, ddp_config=ddp_config, module=self.moe_layer_base)
        # self.model_base = DistributedDataParallel(config=self.config_base, ddp_config=ddp_config, module=model_base)

    def __del__(self):
        torch.distributed.barrier()
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def dispatcher_test(self):
        token_dispatcher = self.token_dispatcher
        token_dispatcher.training = True
        # TODO: Find why setting manual seed can cause the test to fail
        # Manual seed to differentiate input data for each rank
        rank = torch.distributed.get_rank()
        # torch.manual_seed(1000 + rank)
        # random.seed(1000 + rank)
        self.micro_batch_size = 1
        self.sequence_length = 128
        hidden_states = torch.zeros((self.micro_batch_size, self.sequence_length, self.config.hidden_size), dtype=torch.float32)
        # Permute and then unpermute data are supposed to restore original data
        routing_map = torch.zeros((self.micro_batch_size * self.sequence_length, self.num_moe_experts), dtype=torch.bool)
        for i in range(self.sequence_length):
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
        probs = torch.ones((self.micro_batch_size * self.sequence_length, self.num_moe_experts), dtype=torch.float32) / self.config.moe_router_topk
        hidden_states = hidden_states.cuda()
        ans = hidden_states
        hidden_states.requires_grad = True
        routing_map = routing_map.cuda()
        probs = probs.cuda()

        print(f"rank {rank} hidden_states {hidden_states[0, :, 0]}")

        (permuted_local_hidden_states, tokens_per_expert) = (
            token_dispatcher.token_permutation(hidden_states, probs, routing_map)
        )

        print(f"rank {rank} permuted_local_hidden_states {permuted_local_hidden_states[:, 0]}")

        restored_hidden_states, restored_bias = token_dispatcher.token_unpermutation(
            permuted_local_hidden_states
        )

        torch.cuda.current_stream().synchronize()
        print(f"rank {rank} restored_hidden_states {restored_hidden_states[0, :, 0]}")

        # reduce across TP rank equals to multiply data by a scale of ETP
        scale = self.config.expert_tensor_parallel_size
        restored_hidden_states = restored_hidden_states / scale

        assert torch.allclose(
            restored_hidden_states, ans
        ), "Restored hidden states do not match original hidden states"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, hidden_states)

        print(f"rank {rank} hidden_states.grad {hidden_states.grad[0, :, 0]}")
        print(f"rank {rank} ans {ans.grad[0, :, 0]}")
        assert torch.allclose(
            hidden_states.grad, ans
        ), "Restored hidden states do not match original hidden states"


    @pytest.mark.internal
    def perf_test(self):
        rank = torch.distributed.get_rank()
        hidden_states = torch.randn((self.micro_batch_size, self.sequence_length, self.config.hidden_size), dtype=torch.bfloat16).cuda()
        hidden_states_base = hidden_states.clone()
        hidden_states.requires_grad = True

        start_time = torch.cuda.Event(enable_timing=True)
        forward_end_time = torch.cuda.Event(enable_timing=True)
        backward_end_time = torch.cuda.Event(enable_timing=True)
        
        # gather time across all ranks
        time_buf = torch.zeros(2, dtype=torch.float32).cuda()

        # warmup
        warmup_iter = 30
        self.model.zero_grad_buffer()
        with self.model.no_sync():
            for i in range(warmup_iter):
                output, _ = self.model(hidden_states)
                torch.autograd.backward(output, hidden_states)
        self.model.zero_grad_buffer()
        self.moe_layer.training = True

        print(f"rank {rank} DBEP warmup done")

        prof = None
        if output_dir is not None:
            prof = torch.profiler.profile(
                schedule=None,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir, worker_name=f"dbep-{rank}"),
                # record_shapes=True,
                # with_stack=True)
                record_shapes=False,
                with_stack=False)
            prof.start()
            prof.step()

        # global synchronize
        torch.distributed.barrier()
        torch.cuda.synchronize()

        start_time.record()
        output, _ = self.model(hidden_states)
        forward_end_time.record()

        torch.autograd.backward(output, hidden_states)
        backward_end_time.record()
        torch.cuda.synchronize()
        if prof is not None:
            prof.stop()
        forward_time = start_time.elapsed_time(forward_end_time)
        backward_time = forward_end_time.elapsed_time(backward_end_time)
        print(f"rank {rank} DBEP forward time: {forward_time} ms")
        print(f"rank {rank} DBEP backward time: {backward_time} ms")
        print(f"rank {rank} DBEP total time: {forward_time + backward_time} ms")
        print(f"rank {rank} DBEP dispatch time: {self.moe_layer.dispatch_time} ms")
        print(f"rank {rank} DBEP combine time: {self.moe_layer.combine_time} ms")
        print(f"rank {rank} DBEP expert computation time: {self.moe_layer.expert_computation_time} ms")
        print(f"rank {rank} DBEP output_splits_dbep {self.token_dispatcher.output_splits_dbep}")
        print(f"rank {rank} DBEP num_global_tokens_per_local_replica {self.token_dispatcher.num_global_tokens_per_local_replica}")
        # print(f"rank {rank} DBEP time0-1: {self.token_dispatcher.time_0.elapsed_time(self.token_dispatcher.time_1)} ms")
        # print(f"rank {rank} DBEP time1-2: {self.token_dispatcher.time_1.elapsed_time(self.token_dispatcher.time_2)} ms")
        
        time_buf[0] = forward_time + backward_time

        hidden_states_base.requires_grad = True
        start_time = torch.cuda.Event(enable_timing=True)
        forward_end_time = torch.cuda.Event(enable_timing=True)
        backward_end_time = torch.cuda.Event(enable_timing=True)
        
        self.model_base.zero_grad_buffer()
        with self.model_base.no_sync():
            for i in range(warmup_iter):
                output_base, _ = self.model_base(hidden_states_base)
                torch.autograd.backward(output_base, hidden_states_base)
        
        print(f"rank {rank} Base warmup done")
                
        self.model_base.zero_grad_buffer()
        if output_dir is not None:
            prof = torch.profiler.profile(
                schedule=None,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir, worker_name=f"base-{rank}"),
                # record_shapes=True,
                # with_stack=True)
                record_shapes=False,
                with_stack=False)
            prof.start()
            prof.step()

        torch.distributed.barrier()
        torch.cuda.synchronize()

        start_time.record()
        output_base, _ = self.model_base(hidden_states_base)

        forward_end_time.record()
        torch.autograd.backward(output_base, hidden_states_base)
        backward_end_time.record()
        torch.cuda.synchronize()
        if prof is not None:
            prof.stop()

        forward_time = start_time.elapsed_time(forward_end_time)
        backward_time = forward_end_time.elapsed_time(backward_end_time)
        print(f"rank {rank} Base forward time: {forward_time} ms")
        print(f"rank {rank} Base backward time: {backward_time} ms")
        print(f"rank {rank} Base total time: {forward_time + backward_time} ms")
        print(f"rank {rank} Base dispatch time: {self.moe_layer_base.dispatch_time} ms")
        print(f"rank {rank} Base combine time: {self.moe_layer_base.combine_time} ms")
        print(f"rank {rank} Base expert computation time: {self.moe_layer_base.expert_computation_time} ms")
        print(f"rank {rank} Base output_splits {self.token_dispatcher_base.output_splits}")

        time_buf[1] = forward_time + backward_time

        output_time_bufs = [torch.zeros_like(time_buf).cuda() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_time_bufs, time_buf)
        torch.cuda.synchronize()
        if rank == 0:
            # max time across all ranks
            max_time = torch.zeros(2, dtype=torch.float32).cuda()
            for output_time_buf in output_time_bufs:
                max_time = torch.max(max_time, output_time_buf)
            print(f"rank {rank} DBEP max total time: {max_time[0]} ms")
            print(f"rank {rank} Base max total time: {max_time[1]} ms")


    @pytest.mark.internal
    def dispatcher_perf_test(self):
        rank = torch.distributed.get_rank()
        hidden_states = torch.randn((self.micro_batch_size, self.sequence_length, self.config.hidden_size), dtype=torch.bfloat16, requires_grad=True).cuda()
        hidden_states_base = hidden_states.clone()

        with torch.no_grad():
            probs, routing_map = self.moe_layer.router(hidden_states)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        time_0 = torch.cuda.Event(enable_timing=True)
        time_1 = torch.cuda.Event(enable_timing=True)
        time_2 = torch.cuda.Event(enable_timing=True)
        time_3 = torch.cuda.Event(enable_timing=True)
        
        # gather time across all ranks
        time_buf = torch.zeros(2, dtype=torch.float32).cuda()

        # warmup
        warmup_iter = 50
        with self.model.no_sync():
            for i in range(warmup_iter):
                optimizer.zero_grad()
                dispatched_input, tokens_per_expert = self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
                restored_hidden_states, restored_bias = self.token_dispatcher.token_unpermutation(dispatched_input)

                torch.autograd.backward(restored_hidden_states, hidden_states)
        self.moe_layer.training = True

        print(f"rank {rank} DBEP warmup done")

        prof = None
        if output_dir is not None:
            prof = torch.profiler.profile(
                schedule=None,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir, worker_name=f"dbep-{rank}"),
                # record_shapes=True,
                # with_stack=True)
                record_shapes=False,
                with_stack=False)
            prof.start()
            prof.step()

        torch.distributed.barrier()
        torch.cuda.synchronize()
        time_0.record()
        dispatched_input, tokens_per_expert = self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
        time_1.record()
        restored_hidden_states, restored_bias = self.token_dispatcher.token_unpermutation(dispatched_input)

        time_2.record()
        torch.autograd.backward(restored_hidden_states, hidden_states)
        time_3.record()

        torch.cuda.synchronize()
        if prof is not None:
            prof.stop()
        print(f"rank {rank} DBEP dispatch time: {time_0.elapsed_time(time_1)} ms")
        print(f"rank {rank} DBEP combine time: {time_1.elapsed_time(time_2)} ms")
        print(f"rank {rank} DBEP backward time: {time_2.elapsed_time(time_3)} ms")
        print(f"rank {rank} DBEP output_splits_dbep {self.token_dispatcher.output_splits_dbep}")
        
        time_buf[0] = time_0.elapsed_time(time_3)
        print(f"rank {rank} DBEP total time: {time_buf[0]} ms")

        optimizer_base = torch.optim.Adam(self.model_base.parameters(), lr=0.001)
        
        with torch.no_grad():
            probs_base, routing_map_base = self.moe_layer_base.router(hidden_states_base)
        time_0 = torch.cuda.Event(enable_timing=True)
        time_1 = torch.cuda.Event(enable_timing=True)
        time_2 = torch.cuda.Event(enable_timing=True)
        time_3 = torch.cuda.Event(enable_timing=True)
        
        with self.model_base.no_sync():
            for i in range(warmup_iter):
                optimizer_base.zero_grad()
                dispatched_input_base, tokens_per_expert_base = self.token_dispatcher_base.token_permutation(hidden_states_base, probs_base, routing_map_base)
                restored_hidden_states_base, restored_bias_base = self.token_dispatcher_base.token_unpermutation(dispatched_input_base)

                torch.autograd.backward(restored_hidden_states_base, hidden_states_base)
        
        print(f"rank {rank} Base warmup done")
                
        if output_dir is not None:
            prof = torch.profiler.profile(
                schedule=None,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir, worker_name=f"base-{rank}"),
                # record_shapes=True,
                # with_stack=True)
                record_shapes=False,
                with_stack=False)
            prof.start()
            prof.step()
        
        torch.distributed.barrier()
        torch.cuda.synchronize()

        time_0.record()
        dispatched_input_base, tokens_per_expert_base = self.token_dispatcher_base.token_permutation(hidden_states_base, probs_base, routing_map_base)
        time_1.record()
        restored_hidden_states_base, restored_bias_base = self.token_dispatcher_base.token_unpermutation(dispatched_input_base)
        time_2.record()
        torch.autograd.backward(restored_hidden_states_base, hidden_states_base)
        time_3.record()

        torch.cuda.synchronize()
        if prof is not None:
            prof.stop()

        print(f"rank {rank} Base dispatch time: {time_0.elapsed_time(time_1)} ms")
        print(f"rank {rank} Base combine time: {time_1.elapsed_time(time_2)} ms")
        print(f"rank {rank} Base backward time: {time_2.elapsed_time(time_3)} ms")
        print(f"rank {rank} Base total time: {time_0.elapsed_time(time_3)} ms")

        time_buf[1] = time_0.elapsed_time(time_3)

        output_time_bufs = [torch.zeros_like(time_buf).cuda() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_time_bufs, time_buf)
        torch.cuda.synchronize()
        if rank == 0:
            # max time across all ranks
            max_time = torch.zeros(2, dtype=torch.float32).cuda()
            for output_time_buf in output_time_bufs:
                max_time = torch.max(max_time, output_time_buf)
            print(f"rank {rank} DBEP max total time: {max_time[0]} ms")
            print(f"rank {rank} Base max total time: {max_time[1]} ms")

class TestDBEPTokenDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size, ep_size, pp_size, dbep_multiplier", [(1, 1, 2, 2), (1, 2, 1, 2), (1, 1, 1, 2), (1, 1, 1, 4)])
    @pytest.mark.parametrize("num_moe_experts, num_dbep_experts", [(8, 8), (8, 4)])
    @pytest.mark.parametrize("moe_router_topk", [2])
    @pytest.mark.parametrize("permute_fusion", [False])
    @pytest.mark.parametrize("output_dir", [None])
    def test_forward_backward(self, tp_size, ep_size, pp_size, dbep_multiplier, num_moe_experts, num_dbep_experts, moe_router_topk, permute_fusion, output_dir):
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
            output_dir=output_dir,
        )

        # container.dispatcher_test()
        container.perf_test()
        # container.dispatcher_perf_test()

def signal_handler(signum, frame):
    print(f"Signal handler called with signal {signum}")
    stack_trace = traceback.extract_stack(frame)
    print("".join(traceback.format_list(stack_trace)))
    sys.exit(0)

def dump_threads_stacks():
    # dump the stacktrace of all threads
    with open(f'/workspace/userdata/logs/test/dbep/stacktrace-{pid}.log', 'w') as f:
        for thread_id, stack in sys._current_frames().items():
            f.write(f"Thread ID: {thread_id}\n")
            for filename, lineno, name, line in traceback.extract_stack(stack):
                f.write(f"  File: {filename}, line {lineno}, in {name}\n")
                f.write(f"    {line.strip()}\n")
            f.write("\n")
            f.write("-" * 80 + "\n")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    faulthandler.enable()  # 默认捕获 SIGSEGV、SIGFPE 等信号
    # pytest.main([__file__])
    # exit(0)
    args = sys.argv[1:]
    # print(f"args: {args}")
    print("PID: ", os.getpid())
    pid = os.getpid()
    threading.Timer(60, dump_threads_stacks).start()
    if len(args) > 0:
        ep_size = int(args[0])
        pp_size = int(args[1])
        dbep_multiplier = int(args[2])
        num_moe_experts = int(args[3])
        num_dbep_experts = int(args[4])
        moe_router_topk = int(args[5])
        output_dir = args[6]
    else:
        ep_size = 2
        pp_size = 1
        dbep_multiplier = 2
        num_moe_experts = 8
        num_dbep_experts = 4
        moe_router_topk = 2
        output_dir = None
    test = TestDBEPTokenDispatcher()
    test.setup_method(None)
    # threading.Timer(60, dump_threads_stacks).start()
    
    try:
        test.test_forward_backward(
            tp_size=1,
            ep_size=ep_size,
            pp_size=pp_size,
            dbep_multiplier=dbep_multiplier,
            num_moe_experts=num_moe_experts,
            num_dbep_experts=num_dbep_experts,
            moe_router_topk=moe_router_topk,
            permute_fusion=False,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
    # catch sigterm
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        traceback.print_exc()
    test.teardown_method(None)

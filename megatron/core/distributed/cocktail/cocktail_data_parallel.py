# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import itertools
import logging
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ... import parallel_state
from ...config_logger import has_config_logger_enabled, log_config_to_disk
from ...fp8_utils import is_float8tensor
from ...transformer.cuda_graphs import is_graph_capturing
from ...transformer.moe.moe_layer import MoELayer
from ...transformer.transformer_config import TransformerConfig
from ...utils import log_single_rank
from ..data_parallel_base import _BaseDataParallel
from ..distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets

logger = logging.getLogger(__name__)


class CocktailDataParallel(_BaseDataParallel):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).

    Args:
        config: Transformer config object.
        ddp_config: DistributedDataParallel config object.
        module: Underlying model.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket _if_ overlap_grad_reduce is True and pp_rank is 0.

    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
    ):
        super().__init__(config=config, module=module)
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.module = module

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(
                40000000, 1000000 * parallel_state.get_data_parallel_world_size()
            )
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        self.ddp_config = ddp_config
        log_single_rank(
            logger,
            logging.INFO,
            f'Setting up DistributedDataParallel with config {self.ddp_config}',
        )
        
        assert self.ddp_config.num_distributed_optimizer_instances == 1

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size
        if parallel_state.get_pipeline_model_parallel_rank() > 0:
            self.bucket_size = None
        if disable_bucketing:
            self.bucket_size = None
            
        assert (not config.calculate_per_token_loss), "Cannot use per-token loss with cocktail DDP!"
        assert (not self.ddp_config.average_in_collective), "Cannot use average_in_collective with cocktail DDP!"

        # Case 2: average_in_collective=False
        # - Both expert and non-expert parameters:
        #   1. Scale gradients by 1/dp_size before reduction
        #   2. Do sum reduction across data parallel ranks
        #   3. Final result is scaled by 1/dp_size as desired
        data_parallel_world_size = parallel_state.get_data_parallel_world_size(
            with_context_parallel=True
        )

        # Get expert placement
        use_dbep = not (config.dbep_multiplier is None)
        self.use_dbep = use_dbep
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        self.num_global_experts = config.num_moe_experts
        self.num_local_experts = config.num_moe_experts // self.expert_parallel_size
        if use_dbep:
            assert config.num_dbep_experts is not None
            assert parallel_state.get_tensor_model_parallel_world_size() == 1, "DBEP does not support tensor parallelism!"
            self.num_layers = config.num_layers
            self.num_dbep_experts = config.num_dbep_experts
            self.num_local_dbep_experts = self.num_dbep_experts // self.expert_parallel_size
            self.dbep_multiplier = config.dbep_multiplier
            self.dbep_size = self.expert_parallel_size * self.dbep_multiplier
            self.dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True, partial_data_parallel=True)
            self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            self.dbep_rank = parallel_state.get_dbep_rank()
            self.dbep_group_gloo = parallel_state.get_dbep_group_gloo()
            num_dbep_groups = data_parallel_world_size // self.dbep_size
            self.global_rank = torch.distributed.get_rank()
            self.dbep_group_id = parallel_state.get_dbep_group_id() % num_dbep_groups
            self.dbep_alpha_local_gpu = config.dbep_alpha_local_gpu
            self.dbep_alpha_local_node = config.dbep_alpha_local_node
            
            assert self.num_dbep_experts % self.expert_parallel_size == 0
            assert self.num_dbep_experts > 1 and self.num_dbep_experts <= self.num_global_experts
            # assert self.dbep_multiplier > 1
            assert self.num_local_dbep_experts > 1
            assert data_parallel_world_size % self.dbep_size == 0

            import sys
            from pathlib import Path
            root_dir = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
            sys.path.insert(0, str(root_dir) + '/bin/lp')
            from expert_lb import ExpertLB

            self.dbep_instances = {}

            local_layers = []
            for name, module in self.module.named_modules():
                if type(module) is MoELayer and hasattr(module, 'layer_number'):
                    layer_number = getattr(module, 'layer_number')
                    local_layers.append(layer_number)

            for layer_number in local_layers:
                dbep_instance = ExpertLB(num_experts=self.num_dbep_experts, world_size=self.dbep_size, num_replicas_per_gpu=self.num_local_dbep_experts, rank=self.global_rank, num_gpu_per_node=8, alpha_local_gpu=self.dbep_alpha_local_gpu, alpha_local_node=self.dbep_alpha_local_node)
                self.dbep_instances[layer_number] = dbep_instance
            parallel_state.set_dbep_instances(self.dbep_instances)

            # initialize expert placement
            # generate random expert placement for all gpus on gpu 0
            broadcast_buf = torch.zeros((self.num_layers, num_dbep_groups, self.num_dbep_experts + self.dbep_size * self.num_local_dbep_experts * 2), dtype=torch.int32)
            dbep_instance_0 = list(self.dbep_instances.values())[0]
        
            # print(f"rank {self.global_rank} num_layers {self.num_layers} local_layers {local_layers} num_dbep_groups {num_dbep_groups} dbep_group_id {self.dbep_group_id} dbep_size {self.dbep_size} num_local_dbep_experts {self.num_local_dbep_experts} num_local_experts {self.num_local_experts}")
            if self.global_rank == 0:
                for layer_id in range(self.num_layers):
                    for dbep_group_id in range(num_dbep_groups):
                        expert_num_replicas = broadcast_buf[layer_id, dbep_group_id, 0:self.num_dbep_experts].view(-1)
                        expert_replicas = broadcast_buf[layer_id, dbep_group_id, self.num_dbep_experts:self.num_dbep_experts + self.dbep_size * self.num_local_dbep_experts].view(-1)
                        replica_experts = broadcast_buf[layer_id, dbep_group_id, self.num_dbep_experts + self.dbep_size * self.num_local_dbep_experts:].view(-1)
                        
                        # dbep_instance_0.gen_random_expert_placement(expert_num_replicas, expert_replicas, replica_experts)
                        dbep_instance_0.gen_cayley_expert_placement(expert_num_replicas, expert_replicas, replica_experts)
                        print(f"expert_num_replicas {expert_num_replicas}")
                        print(f"expert_replicas {expert_replicas}")
                        print(f"replica_experts {replica_experts}")

            # broadcast expert placement
            broadcast_buf = broadcast_buf.to(device=torch.cuda.current_device())
            torch.distributed.broadcast(
                broadcast_buf,
                src=0,
            )
            broadcast_buf = broadcast_buf.to(device=torch.device("cpu"))
            # config local dbep instances
            for layer_number in local_layers:
                layer_id = layer_number - 1
                expert_num_replicas = broadcast_buf[layer_id, self.dbep_group_id, 0:self.num_dbep_experts].view(-1)
                expert_replicas = broadcast_buf[layer_id, self.dbep_group_id, self.num_dbep_experts:self.num_dbep_experts + self.dbep_size * self.num_local_dbep_experts].view(-1)
                self.dbep_instances[layer_number].set_expert_placement(expert_num_replicas, expert_replicas)

            dbep_expert_dp_ranks = {}

            local_expert_to_global = {}
            for layer_id in range(self.num_layers):
                layer_number = layer_id + 1
                # map local expert to global
                local_expert_to_global[layer_number] = broadcast_buf[layer_id, self.dbep_group_id, self.num_dbep_experts + (self.dbep_size + self.dbep_rank) * self.num_local_dbep_experts:self.num_dbep_experts + (self.dbep_size + self.dbep_rank + 1) * self.num_local_dbep_experts].view(-1).tolist()
                pp_rank = layer_id // ((self.num_layers + self.pp_size - 1) // self.pp_size)
                
                # create all dbep_dp_groups
                for dbep_group_id in range(num_dbep_groups):
                    offset = self.num_dbep_experts
                    for expert in range(self.num_dbep_experts):
                        expert_num_replica = broadcast_buf[layer_id, dbep_group_id, expert].item()
                        ranks = dbep_expert_dp_ranks.get((layer_number, expert), [])
                        for i in range(expert_num_replica):
                            replica = broadcast_buf[layer_id, dbep_group_id, offset + i].item()
                            rank = pp_rank * data_parallel_world_size + dbep_group_id * self.dbep_size + replica // self.num_local_dbep_experts
                            ranks.append(rank)
                        dbep_expert_dp_ranks[(layer_number, expert)] = ranks
                        offset += expert_num_replica
            
            parallel_state.create_dbep_dp_groups(dbep_expert_dp_ranks)
            # print(f"rank {self.global_rank} dbep_expert_dp_ranks {dbep_expert_dp_ranks}")
            del broadcast_buf

        self.param_to_bucket_group = {}

        # Group parameters by their gradient type.
        param_to_name = {}
        dense_params = []
        expert_parallel_params = []
        self.params_with_grad = []
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            # Track params with grad to enable direct setting
            # of param.grad_added_to_main_grad
            self.params_with_grad.append(param)

            param.grad_added_to_main_grad = False
            param_to_name[param] = name

            if getattr(param, 'allreduce', True):
                dense_params.append(param)
            else:
                local_expert_id = getattr(param, 'local_expert_id', -1)
                assert local_expert_id >= 0
                if use_dbep and local_expert_id >= self.num_local_experts - self.num_local_dbep_experts:
                    pass
                else:
                    expert_parallel_params.append(param)
        # create dbep parameter groups
        dbep_params = {}
        if use_dbep:
            for _, module in self.module.named_modules():
                if type(module) is MoELayer:
                    layer_number = module.layer_number
                    for name, param in module.named_parameters():
                        if not param.requires_grad:
                            continue
                        if not getattr(param, 'allreduce', True):
                            local_expert_id = getattr(param, 'local_expert_id', -1)
                            if local_expert_id < self.num_local_experts - self.num_local_dbep_experts:
                                continue
                            global_expert_id = local_expert_to_global[layer_number][local_expert_id - self.num_local_experts + self.num_local_dbep_experts]
                            ranks = dbep_expert_dp_ranks[(layer_number, global_expert_id)]
                            setattr(param, 'dbep_ranks', ranks)
                            params = dbep_params.get(ranks, [])
                            params.append(param)
                            dbep_params[ranks] = params
        # print("dbep_params:", dbep_params.keys())
        
        def _allocate_bucket_groups(buffers):
            # In some scenarios, we want to put buckets from different buffers into a group so that
            # their communication can be aggregated. For example, when there are both fp8 buffers
            # and bf16 buffers in the model and vpp is enabled, each model chunk will have an fp8
            # bucket and a bf16 bucket, which doubles the number of communication kernels, and
            # because of the use of CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back
            # communications will prevent the overlap of the communication kernels with computation
            # kernels.
            # If bucketing is explicitly disabled, then put all buckets in a buffer into a single
            # bucket group.
            bucket_groups = partition_buckets(buffers, force_single_bucket_group=disable_bucketing)

            # Set `next_param_gather_bucket_group` for different bucket groups by iterating through
            # buckets in reverse order (since all-gathers happen in reverse order of buckets).
            if self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather:
                num_bucket_groups = len(bucket_groups)
                for i in range(1, num_bucket_groups):
                    bucket_groups[num_bucket_groups - i].next_param_gather_bucket_group = (
                        bucket_groups[num_bucket_groups - i - 1]
                    )

            # Create map from param to bucket group, used in pre_hook.
            for bucket_group in bucket_groups:
                for bucket in bucket_group.buckets:
                    for param in bucket.params_list:
                        self.param_to_bucket_group[param] = bucket_group

            return bucket_groups

        def _allocate_buffers_for_parameters(
            input_params, data_parallel_group, gradient_scaling_factor
        ):
            param_and_grad_dtype_to_params = {}
            param_and_grad_dtype_to_offsets = {}
            param_and_grad_dtype_to_indices = {}

            # Group parameters by their gradient type.
            for param in input_params:
                assert param.requires_grad

                param_dtype = param.dtype
                if is_float8tensor(param):
                    # Currently TE's Float8Tensor is a wrapper of torch.Tensor. It has a "fake"
                    # dtype (usually a higher precision dtype such as bfloat16), but its actual
                    # data is stored in the form of a torch uint8 tensor within the Float8Tensor's
                    # ".data" attribute. Therefore, when creating the param buffer for fp8 params,
                    # it is necessary to use torch.uint8, not the "fake" dtype got from
                    # "param.dtype".
                    param_dtype = torch.uint8
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
                params.append(param)
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

                # Get the index of each param among the params with same dtype, if a param is fp8,
                # use its "fake" high precision dtype to find which params have same dtype with it.
                # For example:
                #     Case 1:
                #         params = [p1(bf16), p2(bf16), p3(bf16), p4(bf16)]
                #         param_and_grad_dtype_to_indices = {
                #             (torch.bfloat16, torch.float32): [0, 1, 2, 3],
                #         }
                #     Case 2:
                #         params = [p1(bf16), p2(fp8), p3(fp8), p4(bf16)]
                #         param_and_grad_dtype_to_indices = {
                #             (torch.bfloat16, torch.float32): [0, 3],
                #             (torch.uint8, torch.float32): [1, 2],
                #         }
                # We need these indices to load a non-native-fp8 checkpoint in native-fp8 mode.
                offset = param_and_grad_dtype_to_offsets.get((param.dtype, grad_dtype), 0)
                param_and_grad_dtype_to_offsets[(param.dtype, grad_dtype)] = offset + 1
                indices = param_and_grad_dtype_to_indices.get((param_dtype, grad_dtype), [])
                indices.append(offset)
                param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)] = indices

            target_gradient_scaling_factor = 1.0 / parallel_state.get_data_parallel_world_size(
                with_context_parallel=True
            )
            if self.ddp_config.average_in_collective:
                # Collective is averaging gradients in collective with data_parallel_group.
                assert (
                    gradient_scaling_factor
                    / torch.distributed.get_world_size(group=data_parallel_group)
                    == target_gradient_scaling_factor
                )
            else:
                assert gradient_scaling_factor == target_gradient_scaling_factor

            # Allocate the grad buffers and map the grads.
            buffers = []
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
                buffers.append(
                    _ParamAndGradBuffer(
                        self.ddp_config,
                        param_dtype,
                        grad_dtype,
                        params,
                        data_parallel_group,
                        self.bucket_size,
                        param_to_name,
                        gradient_scaling_factor,
                        param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)],
                    )
                )
            
            bucket_groups = _allocate_bucket_groups(buffers=buffers)

            return buffers, bucket_groups


        gradient_scaling_factor = 1.0 / data_parallel_world_size
        expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers, self.bucket_groups = _allocate_buffers_for_parameters(
            dense_params,
            parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=True
            ),
            gradient_scaling_factor=gradient_scaling_factor,
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers, self.expert_parallel_bucket_groups = (
            _allocate_buffers_for_parameters(
                expert_parallel_params,
                parallel_state.get_expert_data_parallel_group(),
                gradient_scaling_factor=expert_gradient_scaling_factor,
            )
        )

        self.dbep_buffers = []
        self.dbep_bucket_groups = []
        for ranks, params in sorted(dbep_params.items(), key=lambda x: x[0]):
            dbep_dp_group = parallel_state.get_dbep_dp_group_from_ranks(ranks)
            assert not dbep_dp_group is None
            dbep_buffer, dbep_bucket_group = (
                _allocate_buffers_for_parameters(
                    params,
                    dbep_dp_group,
                    gradient_scaling_factor=expert_gradient_scaling_factor,
                )
            )
            self.dbep_buffers.append(dbep_buffer)
            self.dbep_bucket_groups.append(dbep_bucket_group)

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:
            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_backward_post_hook(param))
                self.grad_accs.append(grad_acc)

        self.use_forward_hook = (
            self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather
        )
        self.remove_forward_pre_hook_handles = {}
        if self.use_forward_hook:
            self.enable_forward_pre_hook()
        self.overlap_param_gather_with_optimizer_step = False

    def enable_forward_pre_hook(self):
        """
        Enable forward pre-hooks needed for param all-gather overlap with forward compute.
        """
        assert self.use_forward_hook
        assert len(self.remove_forward_pre_hook_handles) == 0
        # Register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
                self._make_forward_pre_hook()
            )

    def disable_forward_pre_hook(self, param_sync: bool = True):
        """
        Disable forward pre-hooks needed for param all-gather overlap with forward compute.
        Skip synchronous param all-gather if `param_sync` is False.
        """
        assert self.use_forward_hook
        # De-register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            assert self.remove_forward_pre_hook_handles[module] is not None
            self.remove_forward_pre_hook_handles[module].remove()
            del self.remove_forward_pre_hook_handles[module]
        assert len(self.remove_forward_pre_hook_handles) == 0

        # Force synchronize parameters.
        if param_sync:
            self.start_param_sync(force_sync=True)

    def _make_forward_pre_hook(self):
        """
        Create a forward pre-hook to wait on all-gather handles when necessary (i.e.,
        when a module uses a parameter in a bucket with a still incomplete all-gather).
        """

        def hook(module, *unused):
            assert (
                self.use_forward_hook
            ), "Should use pre-hook only when overlap_param_gather is True"

            if is_graph_capturing():
                return

            # Make sure all parameters in this module have been all-gathered as necessary.
            for param in module.parameters(recurse=False):
                # Skip parameters without an associated buffer (such parameters have a
                # .requires_grad field equal to False).
                if param not in self.param_to_bucket_group:
                    continue
                assert param.requires_grad

                # If aligning param all-gather across pipeline stages, all-gather is dispatched
                # by start_param_sync calls in core/pipeline_parallelism/schedules.py.
                # If overlapping param all-gather with optimizer step, then all-gather has
                # already been dispatched in optimizer step.
                skip_next_bucket_dispatch = (
                    self.ddp_config.align_param_gather
                    or self.overlap_param_gather_with_optimizer_step
                )
                self.param_to_bucket_group[param].finish_param_sync(
                    skip_next_bucket_dispatch=skip_next_bucket_dispatch
                )

        return hook

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """
        Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
        ready (i.e., when all grads in a bucket have been computed in all microbatches
        in a batch).
        """

        def hook(*unused):
            if is_graph_capturing():
                return

            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(param)

        return hook

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups + list(itertools.chain.from_iterable(self.dbep_bucket_groups)):
            bucket_group.is_last_microbatch = False
        try:
            yield
        finally:
            for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups + list(itertools.chain.from_iterable(self.dbep_bucket_groups)):
                bucket_group.is_last_microbatch = True

    def start_param_sync(self, *unused, force_sync: bool = False, force_dispatch: bool = False):
        """
        Initiates param sync (all-gather) communication operations for all model parameters.

        By default, when overlap_param_gather is set to True, dispatches asynchronous communication
        calls; when overlap_param_gather is set to False, calls synchronous communication
        ops. Can override this default behavior using flags below.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings.
            force_dispatch (bool, optional): force dispatch regardless of other settings.
        """
        if not force_sync:
            # If overlapping param AG with optimizer step, AG should not be dispatched again
            # in forward_backward_step.
            if self.overlap_param_gather_with_optimizer_step and not force_dispatch:
                return

        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups + list(itertools.chain.from_iterable(self.dbep_bucket_groups)):
            bucket_group.start_param_sync(force_sync=force_sync)

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups + list(itertools.chain.from_iterable(self.dbep_bucket_groups)):
            bucket_group.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups + list(itertools.chain.from_iterable(self.dbep_bucket_groups)):
            bucket_group.finish_grad_sync()

    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients inside the buffers by `scaling_factor`."""
        for buffer in self.buffers + self.expert_parallel_buffers + list(itertools.chain.from_iterable(self.dbep_buffers)):
            buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.params_with_grad:
            param.grad_added_to_main_grad = False
        for buffer in self.buffers + self.expert_parallel_buffers + list(itertools.chain.from_iterable(self.dbep_buffers)):
            buffer.reset()
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups + list(itertools.chain.from_iterable(self.dbep_bucket_groups)):
            bucket_group.reset()

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, 'allreduce', True)

            if is_expert_parallel:
                dbep_ranks = getattr(param, 'dbep_ranks', None)
                if dbep_ranks:
                    data_parallel_group = parallel_state.get_dbep_dp_group_from_ranks(dbep_ranks)
                else:
                    data_parallel_group = parallel_state.get_expert_data_parallel_group()
            else:
                data_parallel_group = parallel_state.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=True
                )
            torch.distributed.broadcast(
                param.data,
                src=torch.distributed.get_global_rank(data_parallel_group, 0),
                group=data_parallel_group,
            )

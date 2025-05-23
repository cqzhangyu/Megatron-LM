# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import warnings

from typing import Optional
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm

def get_moe_module_spec(
    num_experts: Optional[int] = None,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    mlp = MLPSubmodules(
        linear_fc1=TEColumnParallelLinear,
        linear_fc2=TERowParallelLinear,
    )

    # experts spec
    expert_module = TEGroupedMLP
    expert_submodule = MLPSubmodules(
        linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
    )

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": False}, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
    )
    return moe_module_spec

def get_mlp_module_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False
) -> ModuleSpec:
    """Helper function to get module spec for MLP"""
    if num_experts is not None:
        if num_experts > 1:
            return get_moe_module_spec(num_experts=num_experts)

        if moe_grouped_gemm:
            raise ValueError("moe_grouped_gemm is not supported with num_experts=1")

    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
        ),
    )

def get_bert_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False
):
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if not HAVE_TE:
        raise ImportError(
            "Transformer Engine is not installed. Please use local Bert layer spec instead."
        )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=get_mlp_module_spec(
                num_experts=num_experts,
                moe_grouped_gemm=moe_grouped_gemm
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def __getattr__(name):
    if name == 'bert_layer_with_transformer_engine_spec':
        warnings.warn(
            """Attribute bert_layer_specs.bert_layer_with_transformer_engine_spec is on a
            deprecation track and will be removed in future releases. Please migrate to
            bert_layer_specs.get_bert_layer_with_transformer_engine_spec()."""
        )

        return get_bert_layer_with_transformer_engine_spec()


# Use this spec for an implementation using only modules in megatron core
bert_layer_local_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=LNImpl,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.padding},
            submodules=SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=LNImpl,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
        ),
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
            'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
        },
    ),
)

#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

APP=$1
GPUS_PER_NODE=$2
# Change for multinode config
NUM_NODES=$3
NODE_RANK=$4
MASTER_ADDR=$5
MASTER_PORT=$6
NUM_EXPERTS=$7
MODEL_SIZE=$8
DATASET=$9
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DIR=/workspace/userdata
mkdir -p ${DIR}/checkpoints/${APP}
mkdir -p ${DIR}/logs/${APP}
CHECKPOINT_PATH=${DIR}/checkpoints/${APP}
TENSORBOARD_LOGS_PATH=${DIR}/logs/${APP}
DATA_PATH=${DIR}/data/${DATASET}/mixtral_text_document
EXPERT_DIST_LOG_PATH=${DIR}/logs/${APP}/dist
TOKENIZER_MODEL=${DIR}/data/mixtral-tokenizer.model

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

LOG_FILE=${DIR}/logs/${APP}/output-${NODE_RANK}.log

if [[ $MODEL_SIZE == "2B" ]]; then
GPT_MODEL_ARGS=(
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 32
    --num-attention-heads 32
    --hidden-size 2048
    --ffn-hidden-size 8192
)
fi
if [[ $MODEL_SIZE == "7B" ]]; then
GPT_MODEL_ARGS=(
    --seq-length 4096
    --max-position-embeddings 32768
    # --num-layers 32
    --num-layers 8
    --num-attention-heads 32
    --hidden-size 4096
    --ffn-hidden-size 14336
)
fi

GPT_MODEL_ARGS+=(
    --use-mcore-models
    --disable-bias-linear
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

if [[ $NUM_EXPERTS -gt $WORLD_SIZE ]]; then
    EP_PARALLEL_SIZE=$WORLD_SIZE
else
    EP_PARALLEL_SIZE=$NUM_EXPERTS
fi

if [[ $NUM_EXPERTS -gt 1 ]]; then
MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --expert-model-parallel-size 4
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
)
fi

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --lr 1e-4
    --train-iters 3000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --use-distributed-optimizer
    --sequence-parallel
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH 
    --expert-dist-log-path $EXPERT_DIST_LOG_PATH
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    # --save $CHECKPOINT_PATH
    # --load $CHECKPOINT_PATH
    --save-interval 10000 
    --eval-iters 10
    --eval-interval 1000 
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-queue-size 1
    --log-throughput
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --profile
    --use-pytorch-profiler
    --profile-step-start 1000
    --profile-step-end 1001
)

cd /workspace/userdata/moelb/Megatron-LM
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    > ${LOG_FILE} 2>&1

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
METHOD=${10}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DIR=/workspace/userdata
mkdir -p ${DIR}/checkpoints/${APP}
mkdir -p ${DIR}/logs/${APP}
CHECKPOINT_PATH=${DIR}/checkpoints/${APP}
TENSORBOARD_LOGS_PATH=${DIR}/logs/${APP}
VOCAB_FILE=${DIR}/data/gpt2-vocab.json
MERGE_FILE=${DIR}/data/gpt2-merges.txt
DATA_PATH=${DIR}/data/${DATASET}/gpt_text_document
EXPERT_DIST_LOG_PATH=${DIR}/logs/${APP}/dist

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

LOG_FILE=${DIR}/logs/${APP}/output-${NODE_RANK}.log

if [[ $MODEL_SIZE == "350M" ]]; then
GPT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --micro-batch-size 4 
    --global-batch-size 256 
)
fi
if [[ $MODEL_SIZE == "760M" ]]; then
GPT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1536 
    --num-attention-heads 16 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --micro-batch-size 8 
    --global-batch-size 256
)
fi
if [[ $MODEL_SIZE == "1.3B" ]]; then
GPT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 2048 
    --num-attention-heads 16 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --micro-batch-size 8 
    --global-batch-size 512 
)
fi
if [[ $MODEL_SIZE == "3.2B" ]]; then
GPT_MODEL_ARGS=(
    --num-layers 16 
    --hidden-size 4096 
    --num-attention-heads 32 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --micro-batch-size 4 
    --global-batch-size 256 
)
fi
if [[ $MODEL_SIZE == "4.8B" ]]; then
GPT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 4096 
    --num-attention-heads 32 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --micro-batch-size 4 
    --global-batch-size 512 
)
fi
if [[ $MODEL_SIZE == "6.7B" ]]; then
# GPT_MODEL_ARGS=(
#     --num-layers 4 
#     --hidden-size 16 
#     --num-attention-heads 1 
#     --seq-length 8 
#     --max-position-embeddings 2048 
#     --micro-batch-size 1 
#     --global-batch-size 16 
# )
GPT_MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096 
    --num-attention-heads 32 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --micro-batch-size 1
    --global-batch-size 256 
)
# GPT_MODEL_ARGS=(
#     --num-layers 32 
#     --hidden-size 4096 
#     --num-attention-heads 32 
#     --seq-length 2048 
#     --max-position-embeddings 2048 
#     --micro-batch-size 1
#     --global-batch-size 256 
# )
fi

DBEP_MULTIPLIER=2
PP_SIZE=$NUM_NODES
# EP_SIZE=WORLD_SIZE/PP_SIZE
EP_SIZE=$((WORLD_SIZE/PP_SIZE/DBEP_MULTIPLIER))
if [[ $NUM_EXPERTS -lt $EP_SIZE ]]; then
    EP_SIZE=$NUM_EXPERTS
fi

if [[ $NUM_EXPERTS -gt 1 ]]; then
MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --expert-model-parallel-size ${EP_SIZE}
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-topk 2
    --moe-router-dtype fp32
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.01
    # --moe-expert-capacity-factor 2.4
    # --moe-capacity-factor-iter 20
    # --moe-router-load-balancing-type none
    # --moe-router-enable-expert-bias
    # --moe-router-score-function sigmoid
    # --moe-router-bias-update-rate 0.001
    --moe-token-dispatcher-type alltoall
    # --moe-enable-deepep
    # --moe-token-dispatcher-type flex
)
fi
if [ $METHOD == "dbep" ]; then
MOE_ARGS+=(
    --num-dbep-experts ${NUM_EXPERTS}
    --dbep-multiplier ${DBEP_MULTIPLIER}
    # --dbep-alpha-local-gpu 10
    # --dbep-alpha-local-node 0.5
)
fi

TRAINING_ARGS=(
    --override-opt_param-scheduler
    --disable-bias-linear
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.014
    --exit-duration-in-mins 600
    --lr 1.2e-4
    --min-lr 1.2e-4
    # --min-lr 1.0e-6
    # --lr 2.0e-4
    # --min-lr 2.0e-5
    --lr-decay-style cosine 
    --split 98,2,0
    --weight-decay 0.1 
    --clip-grad 1.0
    --hysteresis 2
    --num-workers 0
    --grad-reduce-in-bf16
    --bf16
    --overlap-grad-reduce
    --overlap-param-gather
    --recompute-activations
    --recompute-granularity selective
    --recompute-modules moe layernorm
    --no-check-for-nan-in-loss-and-grad
    --empty-unused-memory-level 1
)
# there is a bug without overlap_param_gather, so we must enable it

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1 
    --pipeline-model-parallel-size ${PP_SIZE}
    --use-distributed-optimizer
    --distributed-timeout-minutes 5
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    # --expert-dist-log-path $EXPERT_DIST_LOG_PATH
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    # --save $CHECKPOINT_PATH
    # --load $CHECKPOINT_PATH
    --save-interval 10000 
    --train-iters 50
    --eval-iters 10
    --eval-interval 1000 
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-queue-size 1
    --log-throughput
    # --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    # --log-validation-ppl-to-tensorboard
    --profile
    --use-pytorch-profiler
    --profile-step-start 40
    --profile-step-end 41
    --record-memory-history
    --memory-snapshot-path memory-snapshot-${NODE_RANK}.pickle
)

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL,NET
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/userdata/moelb/Megatron-LM

# rm ${TENSORBOARD_LOGS_PATH}/events.out.tfevents.*
rm ${TENSORBOARD_LOGS_PATH}/*.trace.json
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    > ${LOG_FILE} 2>&1

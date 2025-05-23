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
VOCAB_FILE=${DIR}/data/bert-large-uncased-vocab.txt
DATA_PATH=${DIR}/data/${DATASET}/bert_text_sentence
EXPERT_DIST_LOG_PATH=${DIR}/logs/${APP}/dist-${NODE_RANK}.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

LOG_FILE=${DIR}/logs/${APP}/output.log

if [[ $MODEL_SIZE == "350M" ]]; then
BERT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --micro-batch-size 8 
    --global-batch-size 256 
)
fi
if [[ $MODEL_SIZE == "760M" ]]; then
BERT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1536 
    --num-attention-heads 16 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --micro-batch-size 4 
    --global-batch-size 512 
)
fi
if [[ $MODEL_SIZE == "1.3B" ]]; then
BERT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 2048 
    --num-attention-heads 16 
    --seq-length 2048 
    --max-position-embeddings 2048 
    --micro-batch-size 4 
    --global-batch-size 512 
)
fi

if [[ $NUM_EXPERTS -gt $WORLD_SIZE ]]; then
    EP_PARALLEL_SIZE=$WORLD_SIZE
else
    EP_PARALLEL_SIZE=$NUM_EXPERTS
fi

if [[ $NUM_EXPERTS -gt 1 ]]; then
MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --expert-model-parallel-size ${EP_PARALLEL_SIZE}
    --moe-grouped-gemm
    --moe-router-topk 2
    --moe-router-dtype fp32
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.01
    # --moe-router-load-balancing-type none
    # --moe-router-enable-expert-bias
    # --moe-router-score-function sigmoid
    # --moe-router-bias-update-rate 0.001
    --moe-token-dispatcher-type alltoall
)
fi

TRAINING_ARGS=(
    --override-opt_param-scheduler
    --disable-bias-linear
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.014
    --exit-duration-in-mins 600
    --train-iters 3000
    --lr 1.2e-4
    --min-lr 1.0e-6
    --lr-decay-style cosine 
    --split 98,2,0
    --weight-decay 0.1 
    --clip-grad 1.0
    --hysteresis 2
    --num-workers 0
    --overlap-grad-reduce
    --overlap-param-gather
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1 
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --expert-dist-log-path $EXPERT_DIST_LOG_PATH
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
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
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_bert.py \
    ${BERT_MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    > ${LOG_FILE} 2>&1

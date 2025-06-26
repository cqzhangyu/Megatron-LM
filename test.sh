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
TEST=$7
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DIR=/workspace/userdata
mkdir -p ${DIR}/checkpoints/${APP}
mkdir -p ${DIR}/logs/${APP}
LOG_FILE=${DIR}/logs/${APP}/output.log

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

PP_SIZE=1
DBEP_MULTIPLIER=2
EP_SIZE=$((WORLD_SIZE/PP_SIZE/DBEP_MULTIPLIER))
NUM_MOE_EXPERTS=32
NUM_DBEP_EXPERTS=32
MOE_ROUTER_TOPK=2
OUTPUT_DIR=${DIR}/logs/${APP}

ARGS=(
    ${EP_SIZE}
    ${PP_SIZE}
    ${DBEP_MULTIPLIER}
    ${NUM_MOE_EXPERTS}
    ${NUM_DBEP_EXPERTS}
    ${MOE_ROUTER_TOPK}
    ${OUTPUT_DIR}
)

# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=COLL,NET
export CUDA_LAUNCH_BLOCKING=1

cd /workspace/userdata/moelb/Megatron-LM
rm ${OUTPUT_DIR}/*.trace.json
torchrun ${DISTRIBUTED_ARGS[@]} tests/unit_tests/transformer/moe/$7 ${ARGS[@]} > ${LOG_FILE} 2>&1

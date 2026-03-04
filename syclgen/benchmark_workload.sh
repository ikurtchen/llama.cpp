#!/bin/bash

################################################################################
# Workload Benchmark Script (Model-Level)
# Description: Run llama-bench across multiple models with various input/output
#              lengths and batch sizes on a specified server.
# Usage:       ./benchmark_workload.sh -s <server_name>
################################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

################################################################################
# Per-Server Model Paths
################################################################################
declare -A SERVER_MODEL_PATHS

# Each entry is a newline-separated list of model GGUF paths on the server.
SERVER_MODEL_PATHS=(
    ["h20"]="/ssd/hf_models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
    ["b60"]="/intel/hf_models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
)

################################################################################
# Per-Server Device & Flash-Attention Configuration
################################################################################
declare -A SERVER_DEVICE_ID
declare -A SERVER_FLASH_ATTENTION

SERVER_DEVICE_ID=(
    ["h20"]="0"
    ["b60"]="0"
)

SERVER_FLASH_ATTENTION=(
    ["h20"]="0"
    ["b60"]="0"
)

################################################################################
# Per-Server Environment Prefix (e.g. oneAPI setvars)
################################################################################
declare -A SERVER_ENV_PREFIX

SERVER_ENV_PREFIX=(
    ["h20"]=""
    ["b60"]=""
)

################################################################################
# Test-Case Parameter Lists
# Adjust these lists to cover the workload scenarios you care about.
################################################################################
INPUT_LENS=(128 256 512 1024)
OUTPUT_LENS=(128 256 512)
BATCH_SIZES=(1 2 4 8)

################################################################################
# Parse Arguments
################################################################################
SERVER=""
ENV_EXPORTS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--server)
            SERVER="$2"
            shift 2
            ;;
        -e|--env)
            export "$2"
            ENV_EXPORTS+=("$2")
            shift 2
            ;;
        --help)
            echo "Usage: $0 -s <server_name> [-e VAR=VALUE ...]"
            echo ""
            echo "Options:"
            echo "  -s, --server <name>   Server name (${!SERVER_MODEL_PATHS[*]})"
            echo "  -e, --env <VAR=VAL>   Environment variable to export (repeatable)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$SERVER" ]]; then
    print_error "Server name is required (-s, --server)"
    echo "Available servers: ${!SERVER_MODEL_PATHS[*]}"
    exit 1
fi

if [[ ! -v SERVER_MODEL_PATHS[$SERVER] ]]; then
    print_error "Unknown server: $SERVER"
    echo "Available servers: ${!SERVER_MODEL_PATHS[*]}"
    exit 1
fi

################################################################################
# Resolve Configuration for the Chosen Server
################################################################################
DEVICE_ID="${SERVER_DEVICE_ID[$SERVER]}"
FLASH_ATTENTION="${SERVER_FLASH_ATTENTION[$SERVER]}"
ENV_PREFIX="${SERVER_ENV_PREFIX[$SERVER]}"

# Read model list into an array (newline-separated)
IFS=$'\n' read -r -d '' -a MODEL_LIST <<< "${SERVER_MODEL_PATHS[$SERVER]}" || true

################################################################################
# Run Benchmarks
################################################################################
TOTAL=0
PASSED=0
FAILED=0

print_info "=========================================="
print_info "Workload Benchmark — Server: $SERVER"
print_info "=========================================="
print_info "Models        : ${#MODEL_LIST[@]}"
print_info "Input lengths : ${INPUT_LENS[*]}"
print_info "Output lengths: ${OUTPUT_LENS[*]}"
print_info "Batch sizes   : ${BATCH_SIZES[*]}"
print_info "Device ID     : $DEVICE_ID"
print_info "Flash Attn    : $FLASH_ATTENTION"
print_info "Env Prefix    : ${ENV_PREFIX:-None}"
print_info "Env Exports   : ${ENV_EXPORTS[*]:-None}"
print_info "=========================================="

for MODEL_PATH in "${MODEL_LIST[@]}"; do
    # Skip empty lines
    [[ -z "$MODEL_PATH" ]] && continue

    MODEL_NAME=$(basename "$MODEL_PATH")
    print_info "------------------------------------------"
    print_info "Model: $MODEL_NAME"
    print_info "Path : $MODEL_PATH"
    print_info "------------------------------------------"

    for P in "${INPUT_LENS[@]}"; do
        for N in "${OUTPUT_LENS[@]}"; do
            for B in "${BATCH_SIZES[@]}"; do
                TOTAL=$((TOTAL + 1))

                CMD="./build/bin/llama-bench -m ${MODEL_PATH} -p ${P} -n ${N} -b ${B} -mg ${DEVICE_ID} -fa ${FLASH_ATTENTION}"

                # Prepend environment exports from -e flags
                env_prefix_extra=""
                for ev in "${ENV_EXPORTS[@]}"; do
                    env_prefix_extra="${env_prefix_extra}export ${ev} && "
                done

                if [[ -n "$ENV_PREFIX" ]]; then
                    CMD="${ENV_PREFIX} ${env_prefix_extra}${CMD}"
                elif [[ -n "$env_prefix_extra" ]]; then
                    CMD="${env_prefix_extra}${CMD}"
                fi

                print_info "[${TOTAL}] p=${P} n=${N} b=${B}"
                print_info "CMD: $CMD"

                if eval "$CMD"; then
                    PASSED=$((PASSED + 1))
                    print_success "[${TOTAL}] PASSED  (p=${P} n=${N} b=${B})"
                else
                    FAILED=$((FAILED + 1))
                    print_error "[${TOTAL}] FAILED  (p=${P} n=${N} b=${B})"
                fi

                echo ""
            done
        done
    done
done

################################################################################
# Summary
################################################################################
print_info "=========================================="
print_info "Workload Benchmark Summary"
print_info "=========================================="
print_info "Total : $TOTAL"
print_success "Passed: $PASSED"
if [[ $FAILED -gt 0 ]]; then
    print_error "Failed: $FAILED"
else
    print_info "Failed: 0"
fi
print_info "=========================================="

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi

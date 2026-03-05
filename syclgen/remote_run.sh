#!/bin/bash

################################################################################
# Remote GPU Server Test Script
# Description: Execute tasks on remote SSH servers with different GPU platforms
# Author: Generated Script
# Date: 2026-02-02
################################################################################

set -e  # Exit on error

# WA
ENABLE_BUILD_HACK=1

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default log directory (can be overridden by command line argument)
LOG_DIR="./logs"

################################################################################
# Server Configuration Dictionary
################################################################################
declare -A SERVER_IPS
declare -A SERVER_PORTS
declare -A SERVER_USERS
declare -A SERVER_PASSWORDS
declare -A SERVER_SSH_KEYS
declare -A SERVER_WORKDIRS
declare -A SERVER_CONTAINERS

# Define server configurations
# Format: SERVER_NAME=IP:USER:PASSWORD:SSH_KEY:WORKDIR
SERVER_IPS=(
    ["h20"]="h20"
    ["b60"]="172.16.114.146"
)

SERVER_PORTS=(
    ["h20"]="22"
    ["b60"]="2332"
)

SERVER_USERS=(
    ["h20"]="ubuntu"
    ["b60"]="root"
)

SERVER_PASSWORDS=(
    ["h20"]=""
    ["b60"]=""
)

# SSH key paths (if empty, will use password authentication)
# Set to the path of private key file (e.g., ~/.ssh/id_rsa)
SERVER_SSH_KEYS=(
    ["h20"]="/localdisk/kurt/tools/gpu/new_id_rsa_h20_ww42"
    ["b60"]="~/.ssh/id_rsa"
)

# Default working directories for each server
# Can be overridden by --workdir argument
SERVER_WORKDIRS=(
    ["h20"]="/ssd/kurt/llama.cpp"
    ["b60"]="/intel/kurt/llama.cpp"
)

SERVER_CONTAINERS=(
    ["h20"]="kurt_nv_pt_2601py3"
    ["b60"]=""
)

################################################################################
# Default Commands Dictionary (per server)
################################################################################
declare -A BUILD_COMMANDS
declare -A UNITTEST_COMMANDS
declare -A RUN_COMMANDS
declare -A ACCURACY_COMMANDS
declare -A BENCHMARK_COMMANDS
declare -A PROFILE_COMMANDS

# Build commands
# lama.cpp:
#   cuda: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda
#   sycl: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md#ii-build-llamacpp
#         ["b60"]="./examples/sycl/build.sh"
BUILD_COMMANDS=(
    ["h20"]="cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j8"
    ["b60"]="source /opt/intel/oneapi/setvars.sh && cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DLLAMA_OPENSSL=OFF && cmake --build build --config Release -j -v"
)

# Unit test commands
# Uses test-backend-ops; kernels are filtered via -o flag
UNITTEST_COMMANDS=(
    ["h20"]="./build/bin/test-backend-ops -b CUDA0"
    ["b60"]="source /opt/intel/oneapi/setvars.sh && ./build/bin/test-backend-ops -b SYCL0"
)

# Run commands
RUN_COMMANDS=(
    ["h20"]="./build/bin/llama-cli -m /ssd/hf_models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf -st -p \"List 3 countries and their capitals.\" -n 256"
    ["b60"]="source /opt/intel/oneapi/setvars.sh && ./build/bin/llama-cli -m /intel/hf_models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf -st -mg 0 -p \"List 3 countries and their capitals.\" -n 256"
)

# Accuracy check commands
ACCURACY_COMMANDS=(
    ["h20"]="echo \"not implemented\""
    ["b60"]="echo \"not implemented\""
)

# Kernel benchmark commands
BENCHMARK_COMMANDS=(
    ["h20"]="./build/bin/test-backend-ops perf -b CUDA0"
    ["b60"]="source /opt/intel/oneapi/setvars.sh && ./build/bin/test-backend-ops perf -b SYCL0"
)

# Profile commands
PROFILE_COMMANDS=(
    ["h20"]="echo \"not implemented\""
    ["b60"]="echo \"not implemented\""
)

# Workload benchmark commands (model-level, delegates to benchmark_workload.sh)
# __WORKLOAD_SCRIPT__ is replaced at runtime with the --workload-script value.
WORKLOAD_COMMANDS=(
    ["h20"]="bash __WORKLOAD_SCRIPT__ -s h20"
    ["b60"]="source /opt/intel/oneapi/setvars.sh && bash __WORKLOAD_SCRIPT__ -s b60"
)

# Default workload script path (relative to workdir on the remote server)
DEFAULT_WORKLOAD_SCRIPT="./syclgen/benchmark_workload.sh"

################################################################################
# Helper Functions
################################################################################

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required Arguments:
    -s, --server <name>      Server name (${!SERVER_IPS[@]})
    -t, --task <name>        Task name(s), comma-separated (upload, download, clean, build, unit, run, accuracy, benchmark, profile, workload, custom)

Optional Arguments:
    -d, --workdir <path>        Remote working directory (overrides server default)
    -c, --container <name>      Docker container name to run task in
    -k, --kernels <names>       Kernel/op names to test, comma-separated (e.g. MUL_MAT,SOFT_MAX,ROPE)
    -e, --env <VAR=VALUE>       Environment variables to export (can be used multiple times)
    -l, --logdir <path>         Log directory path (default: ./logs)
    --workload-script <path>    Path to benchmark_workload.sh on the remote server
                                (default: ./syclgen/benchmark_workload.sh)
    --help                      Show this help message

Task Specific Arguments:
    Upload Task:
        --local <path>          Local file or directory path
        --remote <path>         Remote file or directory path

    Download Task:
        --remote <path>         Remote file or directory path
        --local <path>          Local file or directory path

    Unit Test Task:
        -k, --kernels <names>   Filter test-backend-ops by kernel/op names (comma-separated)
        --command <cmd>         Custom command to override default

    Build/Run/Accuracy/Benchmark/Profile Tasks:
        --command <cmd>         Custom command to override default

    Clean Task:
        Removes the build directory (workdir/build) on the remote server

    Custom Task:
        --command <cmd>         Custom command to execute

Available Servers (with default working directories):
$(for server in "${!SERVER_IPS[@]}"; do
    ssh_key_info=""
    if [[ -n "${SERVER_SSH_KEYS[$server]}" ]]; then
        ssh_key_info=" [SSH Key: ${SERVER_SSH_KEYS[$server]}]"
    else
        ssh_key_info=" [Password Auth]"
    fi
    echo "    - $server (${SERVER_IPS[$server]}) - workdir: ${SERVER_WORKDIRS[$server]}$ssh_key_info"
done | sort)

Examples:
    # Upload local directory to remote server (uses default workdir)
    $0 -s nvidia-a100 -t upload --local ./src --remote /workspace/project/src

    # Upload with custom workdir
    $0 -s nvidia-a100 -d /custom/path -t upload --local ./src --remote /custom/path/src

    # Run build task with default command and default workdir
    $0 -s nvidia-a100 -t build

    # Run multiple tasks in sequence (comma-separated)
    $0 -s nvidia-a100 -t build,run,benchmark

    # Clean build directory then rebuild
    $0 -s b60 -t clean,build

    # Run unit tests for specific kernels
    $0 -s b60 -t unit -k MUL_MAT,SOFT_MAX,ROPE

    # Build then test specific kernels
    $0 -s b60 -t build,unit -k MUL_MAT

    # Run build task with custom command and custom workdir
    $0 -s nvidia-a100 -d /workspace/project -t build --command "make clean && make -j8"

    # Run benchmark in a docker container with custom log directory
    $0 -s nvidia-h100 -t benchmark -c gpu_container -l ./my_logs

    # Run custom command with environment variables
    $0 -s amd-mi250 -d /workspace -t custom --command "python train.py" -e "CUDA_VISIBLE_DEVICES=0" -e "OMP_NUM_THREADS=8"

EOF
}

validate_server() {
    local server="$1"
    if [[ ! -v SERVER_IPS[$server] ]]; then
        print_error "Unknown server: $server"
        print_info "Available servers: ${!SERVER_IPS[*]}"
        exit 1
    fi
}

get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

generate_log_filename() {
    local task="$1"
    local server="$2"
    local timestamp=$(get_timestamp)
    echo "${LOG_DIR}/${server}_${task}_${timestamp}.log"
}

# Check if we need sshpass (for password authentication)
check_sshpass() {
    local server="$1"
    local ssh_key="${SERVER_SSH_KEYS[$server]}"

    # Only need sshpass if using password authentication
    if [[ -z "$ssh_key" ]]; then
        if ! command -v sshpass &> /dev/null; then
            print_error "sshpass is not installed. Please install it first."
            print_info "Ubuntu/Debian: sudo apt-get install sshpass"
            print_info "CentOS/RHEL: sudo yum install sshpass"
            print_info "macOS: brew install hudochenkov/sshpass/sshpass"
            print_info ""
            print_info "Alternatively, configure SSH key authentication for this server."
            exit 1
        fi
    fi
}

# Get SSH command prefix based on auth method
get_ssh_prefix() {
    local server="$1"
    local ip="${SERVER_IPS[$server]}"
    local port="${SERVER_PORTS[$server]}"
    local user="${SERVER_USERS[$server]}"
    local password="${SERVER_PASSWORDS[$server]}"
    local ssh_key="${SERVER_SSH_KEYS[$server]}"

    local ssh_opts="-p ${port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

    if [[ -n "$ssh_key" ]]; then
        # Expand tilde in path
        ssh_key="${ssh_key/#\~/$HOME}"
        echo "ssh $ssh_opts -i $ssh_key ${user}@${ip}"
    else
        echo "sshpass -p \"$password\" ssh $ssh_opts ${user}@${ip}"
    fi
}

# Execute SSH command
execute_ssh() {
    local server="$1"
    local command="$2"
    local ssh_cmd=$(get_ssh_prefix "$server")

    $ssh_cmd "$command"
}

# Execute SSH command and save output to log
execute_ssh_with_log() {
    local server="$1"
    local command="$2"
    local log_file="$3"
    local ssh_cmd=$(get_ssh_prefix "$server")

    print_info "Running command: $command on server: $server"
    print_info "Logging output to: $log_file"

    $ssh_cmd "$command" 2>&1 | tee "$log_file"

    return ${PIPESTATUS[0]}
}

# Rsync with SSH
execute_rsync() {
    local server="$1"
    local source="$2"
    local destination="$3"
    local ip="${SERVER_IPS[$server]}"
    local port="${SERVER_PORTS[$server]}"
    local user="${SERVER_USERS[$server]}"
    local password="${SERVER_PASSWORDS[$server]}"
    local ssh_key="${SERVER_SSH_KEYS[$server]}"

    local ssh_opts="-p ${port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

    if [[ -n "$ssh_key" ]]; then
        # Expand tilde in path
        ssh_key="${ssh_key/#\~/$HOME}"
        rsync -avz --progress -e "ssh $ssh_opts -i $ssh_key" \
            "$source" "${user}@${ip}:${destination}"
    else
        sshpass -p "$password" rsync -avz --progress -e "ssh $ssh_opts" \
            "$source" "${user}@${ip}:${destination}"
    fi
}

# Rsync download (from remote to local)
execute_rsync_download() {
    local server="$1"
    local source="$2"
    local destination="$3"
    local ip="${SERVER_IPS[$server]}"
    local port="${SERVER_PORTS[$server]}"
    local user="${SERVER_USERS[$server]}"
    local password="${SERVER_PASSWORDS[$server]}"
    local ssh_key="${SERVER_SSH_KEYS[$server]}"

    local ssh_opts="-p ${port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

    if [[ -n "$ssh_key" ]]; then
        # Expand tilde in path
        ssh_key="${ssh_key/#\~/$HOME}"
        rsync -avz --progress -e "ssh $ssh_opts -i $ssh_key" \
            "${user}@${ip}:${source}" "$destination"
    else
        sshpass -p "$password" rsync -avz --progress -e "ssh $ssh_opts" \
            "${user}@${ip}:${source}" "$destination"
    fi
}

################################################################################
# Task Functions
################################################################################

task_upload() {
    local server="$1"
    local workdir="$2"
    local local_path="$3"
    local remote_path="$4"

    if [[ -z "$local_path" ]] || [[ -z "$remote_path" ]]; then
        print_error "Upload task requires --local and --remote arguments"
        exit 1
    fi

    print_info "Uploading from $local_path to ${server}:${remote_path}"

    # Create remote directory if needed
    execute_ssh "$server" "mkdir -p $(dirname $remote_path)"

    # Upload using rsync
    execute_rsync "$server" "$local_path" "$remote_path"

    print_success "Upload completed successfully"
}

task_download() {
    local server="$1"
    local workdir="$2"
    local remote_path="$3"
    local local_path="$4"

    if [[ -z "$remote_path" ]] || [[ -z "$local_path" ]]; then
        print_error "Download task requires --remote and --local arguments"
        exit 1
    fi

    # check if the remote_path is a absolution or relative path
    if [[ "$remote_path" != /*  && -n "${workdir}" ]]; then
        remote_path="${workdir}/${remote_path}"
    fi

    print_info "Downloading from ${server}:${remote_path} to $local_path"

    # Create local directory if needed
    mkdir -p "$(dirname $local_path)"

    # Download using rsync
    execute_rsync_download "$server" "$remote_path" "$local_path"

    print_success "Download completed successfully"
}

task_build() {
    local server="$1"
    local workdir="$2"
    local container="$3"
    local custom_cmd="$4"
    local env_vars="$5"

    local cmd="${custom_cmd:-${BUILD_COMMANDS[$server]}}"
    local log_file=$(generate_log_filename "build" "$server")

    print_info "Running build on $server"
    print_info "Working directory: $workdir"

    local full_cmd="cd $workdir && $env_vars $cmd"

    if [[ -n "$container" ]]; then
        full_cmd="docker exec $container bash -c 'cd $workdir && $env_vars $cmd'"
    fi

    execute_ssh_with_log "$server" "$full_cmd" "$log_file"

    if [[ $? -eq 0 ]]; then
        print_success "Build completed successfully"
    else
        print_error "Build failed. Check log: $log_file"
        exit 1
    fi
}

task_clean() {
    local server="$1"
    local workdir="$2"
    local container="$3"

    local build_dir="$workdir/build"

    print_info "Cleaning build directory on $server: $build_dir"

    local full_cmd="rm -rf $build_dir"

    if [[ -n "$container" ]]; then
        full_cmd="docker exec $container bash -c 'rm -rf $build_dir'"
    fi

    execute_ssh "$server" "$full_cmd"

    print_success "Build directory removed: $build_dir"
}

task_unittest() {
    local server="$1"
    local workdir="$2"
    local container="$3"
    local custom_cmd="$4"
    local env_vars="$5"
    local kernels="$6"

    local cmd="${custom_cmd:-${UNITTEST_COMMANDS[$server]}}"

    # Append kernel filter: -o accepts comma-separated names natively
    if [[ -n "$kernels" ]]; then
        cmd="$cmd -o $kernels"
        print_info "Filtering kernels: $kernels"
    fi

    local log_file=$(generate_log_filename "unit" "$server")

    print_info "Running unit tests on $server"
    print_info "Working directory: $workdir"
    print_info "Command: $cmd"

    local full_cmd="cd $workdir && $env_vars $cmd"

    if [[ -n "$container" ]]; then
        full_cmd="docker exec $container bash -c 'cd $workdir && $env_vars $cmd'"
    fi

    execute_ssh_with_log "$server" "$full_cmd" "$log_file"

    if [[ $? -eq 0 ]]; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed. Check log: $log_file"
        exit 1
    fi
}

task_run() {
    local server="$1"
    local workdir="$2"
    local container="$3"
    local custom_cmd="$4"
    local env_vars="$5"

    local cmd="${custom_cmd:-${RUN_COMMANDS[$server]}}"
    local log_file=$(generate_log_filename "run" "$server")

    print_info "Running test on $server"
    print_info "Working directory: $workdir"

    local full_cmd="cd $workdir && $env_vars $cmd"

    if [[ -n "$container" ]]; then
        full_cmd="docker exec $container bash -c 'cd $workdir && $env_vars $cmd'"
    fi

    execute_ssh_with_log "$server" "$full_cmd" "$log_file"

    if [[ $? -eq 0 ]]; then
        print_success "Run completed successfully"
    else
        print_error "Run failed. Check log: $log_file"
        exit 1
    fi
}

task_accuracy() {
    local server="$1"
    local workdir="$2"
    local container="$3"
    local custom_cmd="$4"
    local env_vars="$5"

    local cmd="${custom_cmd:-${ACCURACY_COMMANDS[$server]}}"
    local log_file=$(generate_log_filename "accuracy" "$server")

    print_info "Running accuracy check on $server"
    print_info "Working directory: $workdir"

    local full_cmd="cd $workdir && $env_vars $cmd"

    if [[ -n "$container" ]]; then
        full_cmd="docker exec $container bash -c 'cd $workdir && $env_vars $cmd'"
    fi

    execute_ssh_with_log "$server" "$full_cmd" "$log_file"

    if [[ $? -eq 0 ]]; then
        print_success "Accuracy check completed successfully"
    else
        print_error "Accuracy check failed. Check log: $log_file"
        exit 1
    fi
}

task_benchmark() {
    local server="$1"
    local workdir="$2"
    local container="$3"
    local custom_cmd="$4"
    local env_vars="$5"
    local kernels="$6"

    local cmd="${custom_cmd:-${BENCHMARK_COMMANDS[$server]}}"

    # Append kernel filter: -o accepts comma-separated names natively
    if [[ -n "$kernels" ]]; then
        cmd="$cmd -o $kernels"
        print_info "Filtering kernels: $kernels"
    fi

    local log_file=$(generate_log_filename "benchmark" "$server")

    print_info "Running benchmark on $server"
    print_info "Working directory: $workdir"
    print_info "Command: $cmd"

    local full_cmd="cd $workdir && $env_vars $cmd"

    if [[ -n "$container" ]]; then
        full_cmd="docker exec $container bash -c 'cd $workdir && $env_vars $cmd'"
    fi

    execute_ssh_with_log "$server" "$full_cmd" "$log_file"

    if [[ $? -eq 0 ]]; then
        print_success "Benchmark completed successfully"
    else
        print_error "Benchmark failed. Check log: $log_file"
        exit 1
    fi
}

task_profile() {
    local server="$1"
    local workdir="$2"
    local container="$3"
    local custom_cmd="$4"
    local env_vars="$5"

    local cmd="${custom_cmd:-${PROFILE_COMMANDS[$server]}}"
    local log_file=$(generate_log_filename "profile" "$server")

    print_info "Running profiler on $server"
    print_info "Working directory: $workdir"

    local full_cmd="cd $workdir && $env_vars $cmd"

    if [[ -n "$container" ]]; then
        full_cmd="docker exec $container bash -c 'cd $workdir && $env_vars $cmd'"
    fi

    execute_ssh_with_log "$server" "$full_cmd" "$log_file"

    if [[ $? -eq 0 ]]; then
        print_success "Profiling completed successfully"
    else
        print_error "Profiling failed. Check log: $log_file"
        exit 1
    fi
}

task_workload() {
    local server="$1"
    local workdir="$2"
    local container="$3"
    local custom_cmd="$4"
    local env_vars="$5"
    local workload_script="${6:-$DEFAULT_WORKLOAD_SCRIPT}"

    local cmd="${custom_cmd:-${WORKLOAD_COMMANDS[$server]}}"
    # Substitute the workload script path placeholder
    cmd="${cmd//__WORKLOAD_SCRIPT__/$workload_script}"

    # Forward -e env vars as -e flags to benchmark_workload.sh
    if [[ -n "$env_vars" ]]; then
        local env_flags=""
        while read -r pair; do
            [[ -n "$pair" ]] && env_flags="$env_flags -e $pair"
        done < <(echo "$env_vars" | grep -oP '(?<=export )\S+')
        cmd="$cmd$env_flags"
    fi

    local log_file=$(generate_log_filename "workload" "$server")

    print_info "Running workload benchmark on $server"
    print_info "Working directory: $workdir"
    print_info "Command: $cmd"

    local full_cmd="cd $workdir && $env_vars $cmd"

    if [[ -n "$container" ]]; then
        full_cmd="docker exec $container bash -c 'cd $workdir && $env_vars $cmd'"
    fi

    execute_ssh_with_log "$server" "$full_cmd" "$log_file"

    if [[ $? -eq 0 ]]; then
        print_success "Workload benchmark completed successfully"
    else
        print_error "Workload benchmark failed. Check log: $log_file"
        exit 1
    fi
}

task_custom() {
    local server="$1"
    local workdir="$2"
    local container="$3"
    local custom_cmd="$4"
    local env_vars="$5"

    if [[ -z "$custom_cmd" ]]; then
        print_error "Custom task requires --command argument"
        exit 1
    fi

    local log_file=$(generate_log_filename "custom" "$server")

    print_info "Running custom command on $server"
    print_info "Working directory: $workdir"
    print_info "Command: $custom_cmd"

    local full_cmd="cd $workdir && $env_vars $custom_cmd"

    if [[ -n "$container" ]]; then
        full_cmd="docker exec $container bash -c 'cd $workdir && $env_vars $custom_cmd'"
    fi

    execute_ssh_with_log "$server" "$full_cmd" "$log_file"

    if [[ $? -eq 0 ]]; then
        print_success "Custom command completed successfully"
    else
        print_error "Custom command failed. Check log: $log_file"
        exit 1
    fi
}

################################################################################
# Main Script
################################################################################

main() {
    # Parse command line arguments
    local SERVER=""
    local WORKDIR=""
    local TASK=""
    local CONTAINER=""
    local KERNEL_NAMES=""
    local LOCAL_PATH=""
    local REMOTE_PATH=""
    local CUSTOM_COMMAND=""
    local ENV_VARS=""
    local WORKLOAD_SCRIPT=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--server)
                SERVER="$2"
                shift 2
                ;;
            -d|--workdir)
                WORKDIR="$2"
                shift 2
                ;;
            -t|--task)
                TASK="$2"
                shift 2
                ;;
            -c|--container)
                CONTAINER="$2"
                shift 2
                ;;
            -k|--kernels)
                KERNEL_NAMES="$2"
                shift 2
                ;;
            -l|--logdir)
                LOG_DIR="$2"
                shift 2
                ;;
            --local)
                LOCAL_PATH="$2"
                shift 2
                ;;
            --remote)
                REMOTE_PATH="$2"
                shift 2
                ;;
            --command)
                CUSTOM_COMMAND="$2"
                shift 2
                ;;
            --workload-script)
                WORKLOAD_SCRIPT="$2"
                shift 2
                ;;
            -e|--env)
                if [[ -n "$ENV_VARS" ]]; then
                    ENV_VARS="$ENV_VARS export $2 &&"
                else
                    ENV_VARS="export $2 &&"
                fi
                shift 2
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$SERVER" ]]; then
        print_error "Server name is required (-s, --server)"
        print_usage
        exit 1
    fi

    if [[ -z "$TASK" ]]; then
        print_error "Task name is required (-t, --task)"
        print_usage
        exit 1
    fi

    # Validate server
    validate_server "$SERVER"

    # Use default workdir if not specified
    if [[ -z "$WORKDIR" ]]; then
        WORKDIR="${SERVER_WORKDIRS[$SERVER]}"
        print_info "Using default working directory: $WORKDIR"
    fi

    # Use default container if not specified
    if [[ -z "$CONTAINER" ]]; then
        CONTAINER="${SERVER_CONTAINERS[$SERVER]}"
        print_info "Using default container: $CONTAINER"
    fi

    # Create log directory
    mkdir -p "${LOG_DIR}"

    # Check for sshpass if needed
    check_sshpass "$SERVER"

    # Determine auth method
    local auth_method="Password"
    if [[ -n "${SERVER_SSH_KEYS[$SERVER]}" ]]; then
        auth_method="SSH Key (${SERVER_SSH_KEYS[$SERVER]})"
    fi

    # Hack for build task: we need to always upload and clean before build,
    # so we need to add upload and clean tasks before build.
    if [[ ${ENABLE_BUILD_HACK} -eq 1 ]]; then
        if [[ "$TASK" == "build" ]]; then
            TASK="upload,clean,build"
            print_info "Build hack enabled: automatically adding upload and clean tasks before build"

            # since we add upload task, we need to add local and remote args
            if [[ -z "$LOCAL_PATH" ]]; then
                # since the remote_run.sh is under project_root/syclgen,
                # we can get the parent directory of this script as the default local path.
                # We need to get the absolution path of the parent directory to avoid rsync error.
                LOCAL_PATH="$(dirname $(dirname "$(readlink -f "$0")"))"
                print_info "Using default local path: $LOCAL_PATH"
            fi
            if [[ -z "$REMOTE_PATH" ]]; then
                # the remote directory should be the parent of workdir
                REMOTE_PATH="$(dirname "$WORKDIR")"
                print_info "Using default remote path: $REMOTE_PATH"
            fi
        fi
    fi

    print_info "=========================================="
    print_info "Remote GPU Server Test Script"
    print_info "=========================================="
    print_info "Server: $SERVER (${SERVER_IPS[$SERVER]})"
    print_info "Authentication: $auth_method"
    print_info "Working Directory: $WORKDIR"
    print_info "Task(s): $TASK"
    print_info "Log Directory: $LOG_DIR"
    [[ -n "$CONTAINER" ]] && print_info "Container: $CONTAINER"
    [[ -n "$KERNEL_NAMES" ]] && print_info "Kernels: $KERNEL_NAMES"
    [[ -n "$ENV_VARS" ]] && print_info "Environment: $ENV_VARS"
    print_info "=========================================="

    # Split comma-separated tasks and execute each in order
    IFS=',' read -ra TASKS <<< "$TASK"
    local task_count=${#TASKS[@]}
    local task_index=0

    for current_task in "${TASKS[@]}"; do
        # Trim whitespace
        current_task=$(echo "$current_task" | xargs)
        task_index=$((task_index + 1))

        if [[ $task_count -gt 1 ]]; then
            print_info "------------------------------------------"
            print_info "Running task [$task_index/$task_count]: $current_task"
            print_info "------------------------------------------"
        fi

        case $current_task in
            upload)
                task_upload "$SERVER" "$WORKDIR" "$LOCAL_PATH" "$REMOTE_PATH"
                ;;
            download)
                task_download "$SERVER" "$WORKDIR" "$REMOTE_PATH" "$LOCAL_PATH"
                ;;
            clean)
                task_clean "$SERVER" "$WORKDIR" "$CONTAINER"
                ;;
            build)
                task_build "$SERVER" "$WORKDIR" "$CONTAINER" "$CUSTOM_COMMAND" "$ENV_VARS"
                ;;
            unit)
                task_unittest "$SERVER" "$WORKDIR" "$CONTAINER" "$CUSTOM_COMMAND" "$ENV_VARS" "$KERNEL_NAMES"
                ;;
            run)
                task_run "$SERVER" "$WORKDIR" "$CONTAINER" "$CUSTOM_COMMAND" "$ENV_VARS"
                ;;
            accuracy)
                task_accuracy "$SERVER" "$WORKDIR" "$CONTAINER" "$CUSTOM_COMMAND" "$ENV_VARS"
                ;;
            benchmark)
                task_benchmark "$SERVER" "$WORKDIR" "$CONTAINER" "$CUSTOM_COMMAND" "$ENV_VARS" "$KERNEL_NAMES"
                ;;
            profile)
                task_profile "$SERVER" "$WORKDIR" "$CONTAINER" "$CUSTOM_COMMAND" "$ENV_VARS"
                ;;
            workload)
                task_workload "$SERVER" "$WORKDIR" "$CONTAINER" "$CUSTOM_COMMAND" "$ENV_VARS" "$WORKLOAD_SCRIPT"
                ;;
            custom)
                task_custom "$SERVER" "$WORKDIR" "$CONTAINER" "$CUSTOM_COMMAND" "$ENV_VARS"
                ;;
            *)
                print_error "Unknown task: $current_task"
                print_info "Available tasks: upload, download, clean, build, unit, run, accuracy, benchmark, profile, workload, custom"
                exit 1
                ;;
        esac
    done

    print_info "=========================================="
    print_success "All $task_count task(s) completed successfully!"
    print_info "=========================================="
}

# Run main function
main "$@"

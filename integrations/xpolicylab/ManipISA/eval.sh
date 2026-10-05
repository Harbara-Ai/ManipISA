#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 10 ]]; then
    echo 'Usage: eval.sh RoboDojo TASK CKPT arx_x5 ee SEED POLICY_GPU ENV_GPU POLICY_CONDA ENV_CONDA' >&2
    exit 2
fi
bench=$1 task=$2 checkpoint=$3 embodiment=$4 action=$5 seed=$6
policy_gpu=$7 env_gpu=$8 policy_env=$9 sim_env=${10}
if [[ "$bench" != RoboDojo || "$embodiment" != arx_x5 || "$action" != ee ]]; then
    echo 'This adapter supports RoboDojo / arx_x5 / ee only.' >&2
    exit 2
fi
policy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
xpl_root=$(cd -- "$policy_dir/../.." && pwd)
bench_root=$(cd -- "$xpl_root/.." && pwd)
port=$(bash "$xpl_root/utils/get_free_port.sh")
config="$policy_dir/deploy.yml"
cleanup() {
    if [[ -n "${server_pid:-}" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
(
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$policy_env"
    exec env CUDA_VISIBLE_DEVICES="$policy_gpu" python "$xpl_root/setup_policy_server.py" \
        --config_path "$config" --overrides policy_name=ManipISA bench_name="$bench" \
        task_name="$task" ckpt_name="$checkpoint" env_cfg_type="$embodiment" \
        action_type=ee seed="$seed" host=localhost port="$port"
) &
server_pid=$!
bash "$xpl_root/utils/wait_for_policy_server.sh" localhost "$port" "$server_pid" 'ManipISA policy' 1200
bash "$xpl_root/utils/setup_env_client.sh" \
    "$xpl_root/utils" "$config" "$sim_env" "$port" "$bench" "$task" "$embodiment" \
    ManipISA "ckpt_name=$checkpoint,action_type=ee" "$bench_root" "$seed" "$env_gpu" localhost

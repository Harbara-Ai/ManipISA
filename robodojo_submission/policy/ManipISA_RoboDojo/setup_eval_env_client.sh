#!/bin/bash
# Adapted XPolicyLab entrypoint; local isolation/timeout/cleanup changes. See NOTICE.
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

if [[ "${bench_name}" != RoboDojo || "${env_cfg_type}" != arx_x5 || "${action_type}" != ee ]]; then
    echo 'This policy requires RoboDojo/arx_x5/ee.' >&2
    exit 2
fi
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
    exec bash "${UTILS_DIR}/setup_env_client.sh" "${UTILS_DIR}" "${yaml_file}" \
        "${eval_env_conda_env}" "${policy_server_port}" "${bench_name}" "${task_name}" \
        "${env_cfg_type}" "${policy_name}" "${additional_info}" "${BENCH_ROOT}" \
        "${seed}" "${env_gpu_id}" "${policy_server_ip}"
fi
if [[ "${EVAL_ENV_TYPE:-sim}" != sim ]]; then
    echo 'This submission only supports simulation evaluation.' >&2
    exit 2
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${eval_env_conda_env}"
export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_$$}"
url_host="${policy_server_ip}"
if [[ "${url_host}" == *:* && "${url_host}" != \[*\] ]]; then
    url_host="[${url_host}]"
fi
cd "${BENCH_ROOT}"
attempt=0
while :; do
    set +e
    python -u "${SCRIPT_DIR}/run_eval_client.py" \
        --task_name "${task_name}" --env_cfg_type "${env_cfg_type}" --num_envs 1 \
        --enable_cameras --headless \
        --kit_args ' --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera' \
        --device_id "${env_gpu_id}" --policy_name "${policy_name}" \
        --port "${policy_server_port}" --protocol ws --host "${policy_server_ip}" \
        --policy_server_url "ws://${url_host}:${policy_server_port}" \
        --additional_info "${additional_info}" --seed "${seed}"
    rc=$?
    set -e
    case "$rc" in
        0) exit 0 ;;
        99|134|139)
            attempt=$((attempt + 1))
            if [[ "$attempt" -ge "${ROBODOJO_MAX_BASH_RETRIES:-10}" ]]; then exit "$rc"; fi
            sleep 5 ;;
        *) exit "$rc" ;;
    esac
done

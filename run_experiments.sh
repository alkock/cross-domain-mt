#!/usr/bin/env bash
set -euo pipefail

PROWL_API_KEY="${PROWL_API_KEY:-}"
WORKSPACE="${WORKSPACE:-$HOME/projects/cross-domain-evaluation-of-deep-learning-approaches-for-spatial-data-translation}"
DATA_DIR="${DATA_DIR:-/mnt/scratch/eu2384}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/tmp/experiment_data}"
IMAGE="pix2pix-experiments:latest"

EXPERIMENTS=(
    "exp-heihc-fold:experiments/heihc_fold.py"
    "exp-heihc-full:experiments/heihc_full.py"
    "exp-saropt-fold:experiments/saropt_fold.py"
    "exp-saropt-full:experiments/saropt_full.py"
)

prowl_notify() {
    local event="$1"
    local description="$2"
    local priority="${3:-0}"

    if [[ -z "${PROWL_API_KEY:-}" ]]; then
        return
    fi

    curl -s --max-time 10 "https://api.prowlapp.com/publicapi/add" \
        --data-urlencode "apikey=${PROWL_API_KEY}" \
        --data-urlencode "application=Pix2Pix Training" \
        --data-urlencode "event=${event}" \
        --data-urlencode "description=${description}" \
        --data-urlencode "priority=${priority}" \
        -o /dev/null || echo "[prowl] Notification failed (network error)."
}

cleanup() {
    echo ""
    echo "Cleaning up local data copy at ${LOCAL_DATA_DIR} ..."
    rm -rf "${LOCAL_DATA_DIR}"
    echo "Done."
}
trap cleanup EXIT

echo "Copying data from ${DATA_DIR} to local SSD at ${LOCAL_DATA_DIR} ..."
mkdir -p "${LOCAL_DATA_DIR}"
rsync -a --info=progress2 "${DATA_DIR}/HE/"          "${LOCAL_DATA_DIR}/HE/"
rsync -a --info=progress2 "${DATA_DIR}/IHC/"         "${LOCAL_DATA_DIR}/IHC/"
rsync -a --info=progress2 "${DATA_DIR}/SEN12-splits/" "${LOCAL_DATA_DIR}/SEN12-splits/"
echo "Data copy complete."

echo "Building Docker image: ${IMAGE}"
docker build -t "${IMAGE}" "${WORKSPACE}"

for entry in "${EXPERIMENTS[@]}"; do
    CONTAINER="${entry%%:*}"
    CONFIG="${entry##*:}"
    EXPERIMENT_NAME=$(basename "${CONFIG}" .py)
    LOG_FILE="${WORKSPACE}/logs/${EXPERIMENT_NAME}.log"
    mkdir -p "$(dirname "${LOG_FILE}")"

    echo ""
    echo "========================================"
    echo "Starting: ${EXPERIMENT_NAME}"
    echo "Script:   ${CONFIG}"
    echo "Log:      ${LOG_FILE}"
    echo "========================================"

    prowl_notify "Started: ${EXPERIMENT_NAME}" "Script: ${CONFIG}" -1

    set +e
    docker run --rm \
        --gpus all \
        --shm-size=8g \
        --name "${CONTAINER}" \
        -v "${WORKSPACE}:/workspace" \
        -v "${LOCAL_DATA_DIR}:/workspace/mnt-data" \
        -w /workspace/experiment \
        "${IMAGE}" \
        bash -c "python ${CONFIG} 2>&1" \
        | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[${EXPERIMENT_NAME}] Finished successfully."
        prowl_notify "SUCCESS: ${EXPERIMENT_NAME}" "Training completed. Log: ${LOG_FILE}" 0
    else
        echo "[${EXPERIMENT_NAME}] FAILED with exit code ${EXIT_CODE}."
        prowl_notify "FAILED: ${EXPERIMENT_NAME}" "Exit code ${EXIT_CODE}. Check log: ${LOG_FILE}" 2
        echo "Aborting remaining experiments."
        exit ${EXIT_CODE}
    fi
done

echo ""
echo "Experiments completed successfully."
prowl_notify "All done experiments finished." 0

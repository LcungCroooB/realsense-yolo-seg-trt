#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TARGET="yolo_seg_trt_app"
BUILD_TYPE="Release"
RUN_AFTER_BUILD=0
CLEAN_FIRST=0
RUN_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./scripts/build_apps.sh [options]

Options:
    --run           Run yolo_seg_trt_app after successful build
    --clean         Remove build directory before configuring
    --build-dir D   Build directory (default: ./build)
    --build-type T  CMake build type (default: Release)
    --              Pass remaining args to the executable when used with --run
    -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN_AFTER_BUILD=1
            shift
            ;;
        --clean)
            CLEAN_FIRST=1
            shift
            ;;
        --build-type)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --build-type requires a value"
                exit 1
            fi
            BUILD_TYPE="$2"
            shift 2
            ;;
        --build-dir)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --build-dir requires a value"
                exit 1
            fi
            BUILD_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            RUN_ARGS=("$@")
            break
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "${CLEAN_FIRST}" -eq 1 && -d "${BUILD_DIR}" ]]; then
    echo "[INFO] Removing build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

echo "[INFO] Configuring project"
cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

echo "[INFO] Building target: ${TARGET}"
cmake --build "${BUILD_DIR}" --target "${TARGET}" -j"$(nproc)"

APP_BIN="${BUILD_DIR}/bin/${TARGET}"
if [[ "${RUN_AFTER_BUILD}" -eq 1 ]]; then
    if [[ ! -x "${APP_BIN}" ]]; then
        echo "[ERROR] Executable not found: ${APP_BIN}"
        exit 1
    fi
    echo "[INFO] Running: ${APP_BIN} ${RUN_ARGS[*]}"
    "${APP_BIN}" "${RUN_ARGS[@]}"
fi

echo "[INFO] Done"

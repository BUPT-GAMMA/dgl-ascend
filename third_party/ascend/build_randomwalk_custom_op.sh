#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <randomwalk_gen_dir> <ascend_path> <Debug|Release>" >&2
  exit 1
fi

op_dir="$1"
ascend_path="$2"
build_type="$3"

export ASCEND_HOME_PATH="${ascend_path}"
export ASCEND_CANN_PACKAGE_PATH="${ascend_path}"

if [ -z "${ASCEND_PYTHON_EXECUTABLE:-}" ]; then
  for candidate in \
    "${CONDA_PREFIX:-}/bin/python" \
    "$HOME/miniconda3/envs/dgl_env/bin/python" \
    python3; do
    if command -v "${candidate}" >/dev/null 2>&1 &&
       "${candidate}" -c "import numpy" >/dev/null 2>&1; then
      export ASCEND_PYTHON_EXECUTABLE="${candidate}"
      break
    fi
  done
fi

if [ -z "${ASCEND_PYTHON_EXECUTABLE:-}" ]; then
  echo "Cannot find a Python executable with numpy for Ascend OPP build." >&2
  exit 1
fi

cd "${op_dir}"
bash build.sh -b "${build_type}"

shopt -s nullglob
installers=(build_out/custom_opp_*.run)
if [ "${#installers[@]}" -eq 0 ]; then
  echo "Cannot find generated RandomWalkCustom OPP installer under ${op_dir}/build_out" >&2
  exit 1
fi

chmod +x "${installers[0]}"
"${installers[0]}"

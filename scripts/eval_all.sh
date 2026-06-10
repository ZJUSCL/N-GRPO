#!/usr/bin/env bash
set -euo pipefail

# 导出模型并顺序跑多个数据集评测，结束后自动清理导出模型。

# 支持通过环境变量覆盖默认值，未设置时落回下方默认。
# 优先级：
# 1) MODEL_PATH 直接评测（不做 merge）
# 2) MODEL_ROOT + STEP 走 merge 到 TARGET_DIR
MODEL_PATH="${MODEL_PATH:-models/DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_ROOT="${MODEL_ROOT:-}"
STEP="${STEP:-}"
TARGET_DIR="${TARGET_DIR:-outputs/hf_model}"
CONFIG_DIR="${CONFIG_DIR:-configs}"
CONFIG_NAME="${CONFIG_NAME:-eval_1_5b.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./tmp/tmp}"

# 验证采样参数（控制 pass@k）
N="${N:-32}"                  # 验证每条样本生成次数，决定 pass@k
DO_SAMPLE="${DO_SAMPLE:-true}"        # 是否采样，多样解需要开启；为 false 时 n/温度/采样参数基本无效
TEMPERATURE="${TEMPERATURE:-0.6}"     # 采样温度，do_sample=true 时生效
TOP_P="${TOP_P:-0.95}"                # Top-p
TOP_K="${TOP_K:--1}"                  # Top-k；sglang/vLLM 下 -1 走 engine 默认

LOCAL_DIR=""
ACTOR_PATH=""
MERGED=false

safe_remove_dir() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    echo "[clean] rm -rf ${dir}"
    rm -rf "${dir}"
  fi
}

if [[ -n "${MODEL_ROOT}" && -n "${STEP}" ]]; then
  LOCAL_DIR="${MODEL_ROOT}/global_step_${STEP}/actor"
  ACTOR_PATH="${TARGET_DIR}"
  MERGED=true
  echo "[merge] 导出前清理旧目录：${TARGET_DIR}"
  safe_remove_dir "${TARGET_DIR}"

  echo "[merge] 开始合并模型：${LOCAL_DIR} -> ${TARGET_DIR}"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}"
else
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "[error] 未提供 MODEL_PATH，必须提供 MODEL_ROOT 和 STEP 才能 merge。"
    exit 1
  fi
  ACTOR_PATH="${MODEL_PATH}"
fi

# 数据集名称与 val_files 路径的映射，按顺序执行
declare -a TASKS=(
  "amc23:data/verl_data/amc23/test.parquet"
  "aime24:data/verl_data/aime24/test.parquet"
  "aime25:data/verl_data/aime25/test.parquet"
  "math500:data/verl_data/math_500/test.parquet"
  "gpqa_diamond:data/verl_data/gpqa_diamond/test.parquet"
  # "mmlu_pro:data/verl_data/mmlu_pro/test.parquet"
  # "mmlu_pro_valid:data/verl_data/mmlu_pro/validation.parquet"
)

for item in "${TASKS[@]}"; do
  IFS=":" read -r name data_path <<< "${item}"
  OUT_DIR="${OUTPUT_ROOT}/${name}"

  echo "[eval] 清理旧输出：${OUT_DIR}"
  safe_remove_dir "${OUT_DIR}"
  mkdir -p "${OUT_DIR}"

  # 参数提示
  if [[ "${DO_SAMPLE}" != "true" ]]; then
    echo "[hint] do_sample=false：n/temperature/top_p/top_k 基本无效，等价 greedy；若要 pass@${N} 需改为 true。"
  fi

  echo "[eval] val_kwargs: n=${N}, do_sample=${DO_SAMPLE}, temperature=${TEMPERATURE}, top_p=${TOP_P}, top_k=${TOP_K}"

  echo "[eval] 使用配置 ${CONFIG_NAME}，数据集 ${name} (${data_path}) -> ${OUT_DIR}"
  PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-dir="${CONFIG_DIR}" \
    --config-name="${CONFIG_NAME}" \
    trainer.resume_mode=disable \
    actor_rollout_ref.model.path="${ACTOR_PATH}" \
    trainer.val_only=true \
    trainer.test_freq=1 \
    "data.val_files=[${data_path}]" \
    trainer.validation_data_dir="${OUT_DIR}" \
    actor_rollout_ref.rollout.val_kwargs.n="${N}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${DO_SAMPLE}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${TOP_K}" \
    2>&1 | tee "${OUT_DIR}/eval.log"
done

if [[ "${MERGED}" == "true" ]]; then
  echo "[cleanup] 评测完成，删除导出模型目录：${TARGET_DIR}"
  safe_remove_dir "${TARGET_DIR}"
fi

echo "[done] 全部任务结束，结果位于 ${OUTPUT_ROOT}"

#!/bin/bash
# ============================================================
# PaddleSpeech TTS 安装脚本 — 树莓派 5 (aarch64)
# ============================================================
# 用法:
#   chmod +x install_paddlespeech.sh
#   ./install_paddlespeech.sh
#
# 策略: 轻量安装，只装 TTS 推理所需的最小依赖
#   - 创建 conda 环境 paddlespeech (Python 3.10)
#   - pip 安装 PaddlePaddle (ARM64)
#   - 源码可编辑安装 PaddleSpeech (--no-deps)
#   - 逐步补充 TTS 推理依赖
#   - 首次运行自动下载模型 (~1.5GB)
# ============================================================

set -e

ENV_NAME="paddlespeech"
PYTHON_VER="3.10"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
PADDLE_SPEECH_DIR="$HOME/tts/PaddleSpeech-develop"

echo "=============================================="
echo " PaddleSpeech TTS 安装脚本"
echo " 目标设备: 树莓派 5 (aarch64)"
echo " Conda 环境: $ENV_NAME (Python $PYTHON_VER)"
echo "=============================================="

# ---- 1. 初始化 conda ----
if [ ! -f "$CONDA_SH" ]; then
    echo "[错误] 找不到 conda 配置: $CONDA_SH"
    exit 1
fi
source "$CONDA_SH"

# ---- 2. 创建 conda 环境 ----
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[跳过] conda 环境 '$ENV_NAME' 已存在"
else
    echo "[创建] conda 环境 '$ENV_NAME' (Python $PYTHON_VER)..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VER"
fi

conda activate "$ENV_NAME"
echo "[当前 Python] $(python --version)"

# ---- 3. 安装 PaddlePaddle (ARM64) ----
echo "[安装] PaddlePaddle (ARM64)..."
python -c "import paddle" 2>/dev/null && echo "[跳过] PaddlePaddle 已安装" || {
    pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/cpu/stable.html
}

# ---- 4. 源码可编辑安装 PaddleSpeech (不装重依赖避免 OOM) ----
echo "[安装] PaddleSpeech (editable, no deps)..."
python -c "import paddlespeech" 2>/dev/null && echo "[跳过] PaddleSpeech 已安装" || {
    cd "$PADDLE_SPEECH_DIR"
    pip install -e . --no-deps --no-build-isolation
}

# ---- 5. 逐步安装 TTS 推理所需最小依赖 ----
echo "[安装] TTS 推理依赖..."
pip install pyyaml yacs pypinyin prettytable requests soundfile
pip install tqdm websockets zhon pydantic inflect typeguard jieba
pip install librosa ffmpeg-python rich h5py onnxruntime

# ---- 6. 修补 paddlenlp/aistudio_sdk 兼容问题 ----
echo "[修补] paddlenlp aistudio_sdk 兼容..."
AISTUDIO_FILE="$HOME/miniconda3/envs/$ENV_NAME/lib/python3.10/site-packages/paddlenlp/transformers/aistudio_utils.py"
if grep -q "from aistudio_sdk.hub import download" "$AISTUDIO_FILE" 2>/dev/null; then
    sed -i 's/from aistudio_sdk.hub import download/try:\n    from aistudio_sdk.hub import download\nexcept ImportError:\n    download = None/' "$AISTUDIO_FILE"
    echo "[修补] aistudio_utils.py 已修复"
else
    echo "[跳过] aistudio_utils.py 无需修补"
fi

# ---- 7. 验证 ----
echo ""
echo "=============================================="
echo " 验证安装..."
echo "=============================================="
python -c "import paddle; print(f'[OK] PaddlePaddle {paddle.__version__}')"
python -c "from paddlespeech.cli.tts import TTSExecutor; print('[OK] TTSExecutor 导入成功')"

echo ""
echo "=============================================="
echo " 安装完成！"
echo ""
echo " 首次合成自动下载模型 (约1.5GB)，请耐心等待:"
echo "   conda activate $ENV_NAME"
echo "   python test_paddlespeech_tts.py -t '你好树莓派' --play"
echo "=============================================="

# PaddleSpeech 中文 TTS — 树莓派 5 配置文档

> **设备**: 树莓派 5 Model B, 8GB RAM, aarch64, Debian 13
> **创建日期**: 2026-06-08

---

## 一、方案概述

使用百度 **PaddleSpeech** 离线语音合成引擎，替代 espeak-ng。

| 对比维度 | espeak-ng | PaddleSpeech |
|----------|-----------|-------------|
| 合成方式 | 共振峰拼接 | 神经网络 (FastSpeech2 + HiFiGAN) |
| 声音自然度 | ★☆☆☆☆ 机械感强 | ★★★★☆ 接近真人 |
| 中文支持 | 勉强能听 | 优秀（CSMSC baker 数据集，女声） |
| 模型大小 | ~5MB | ~1.7GB（FastSpeech2 + HiFiGAN + BERT） |
| 推理速度 (RPi5) | 即时 | 首次 ~55s，后续 4~5 秒/句 |
| 离线运行 | ✅ | ✅ |
| **实测状态** | — | ✅ 已在 RPi5 验证通过 |

---

## 二、安装

### 2.1 一键安装（推荐）

```bash
cd ~/tts
chmod +x install_paddlespeech.sh
./install_paddlespeech.sh
```

安装策略: 轻量安装，只装 TTS 推理所需的最小依赖，避免系统 OOM。
耗时约 10-20 分钟。

### 2.2 手动安装（分步）

```bash
# 1. 初始化 conda
source ~/miniconda3/etc/profile.d/conda.sh

# 2. 创建环境
conda create -y -n paddlespeech python=3.10
conda activate paddlespeech

# 3. 安装 PaddlePaddle (ARM64 专用源)
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/cpu/stable.html

# 4. 源码可编辑安装 PaddleSpeech (不装重依赖，避免 OOM)
cd ~/tts/PaddleSpeech-develop
pip install -e . --no-deps --no-build-isolation

# 5. 逐步安装 TTS 推理依赖
pip install pyyaml yacs pypinyin prettytable requests soundfile
pip install tqdm websockets zhon pydantic inflect typeguard jieba
pip install librosa ffmpeg-python rich h5py onnxruntime

# 6. 修补 paddlenlp/aistudio_sdk 兼容（树莓派必须）
sed -i 's/from aistudio_sdk.hub import download/try:\n    from aistudio_sdk.hub import download\nexcept ImportError:\n    download = None/' \
  ~/miniconda3/envs/paddlespeech/lib/python3.10/site-packages/paddlenlp/transformers/aistudio_utils.py
```

---

## 三、使用

### 3.1 激活环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate paddlespeech
```

### 3.2 命令行合成（最快上手）

```bash
# 单句合成
paddlespeech tts --input "你好树莓派" --output hello.wav

# 播放
aplay hello.wav
```

### 3.3 Python 脚本

```bash
# 单句合成 + 播放
python test_paddlespeech_tts.py -t "你好，我是语音助手" --play

# 批量合成测试
python test_paddlespeech_tts.py --batch

# 使用 Python API 方式
python test_paddlespeech_tts.py -m api -t "今天天气不错"
```

### 3.4 在你的项目中集成

```python
from paddlespeech.cli.tts import TTSExecutor

tts = TTSExecutor()
tts(text="前方发现障碍物，请注意避让", output="/tmp/alert.wav")

# 然后用 aplay 播放
import subprocess
subprocess.run(["aplay", "/tmp/alert.wav"])
```

---

## 四、模型管理

### 4.1 模型清单

| 组件 | 大小 | 路径 |
|------|------|------|
| `fastspeech2_csmsc-zh` | ~489MB | `~/.paddlespeech/models/` |
| `hifigan_csmsc-zh` | ~915MB | `~/.paddlespeech/models/` |
| `bert-base-chinese` | ~400MB | `~/.paddlenlp/models/` |
| `G2PWModel_1.1` | ~562MB | `~/.paddlespeech/models/` |

> 模型总计约 **2.3GB**，首次运行自动下载，后续无需联网。
> 英文模型 (fastspeech2_ljspeech / hifigan_ljspeech, ~2.7GB) 已移除，如需恢复请联系 AI。

### 4.2 清理模型释放空间

```bash
# 查看模型占用
du -sh ~/.paddlespeech/models/*
du -sh ~/.paddlenlp/models/*

# 清空所有模型（下次运行会重新下载）
rm -rf ~/.paddlespeech/models/*
```

---

## 五、性能参考（树莓派 5 实测）

| 文本长度 | 合成耗时 | 内存峰值 |
|----------|----------|----------|
| 短句 (~10字) | 3-5 秒 | ~1.5GB |
| 中等 (~30字) | 5-10 秒 | ~2GB |
| 长句 (~60字) | 10-20 秒 | ~3GB |

> 树莓派 5 8GB 内存完全够用。模型下载约需 3-5 分钟（1.5GB）。

---

## 六、重要注意事项

### ⚠️ 避免 OOM（内存溢出）

1. **不要**直接 `pip install paddlespeech`（会触发大量重依赖安装导致系统死机）
2. **必须**使用 `--no-deps` 安装核心，再逐步补充依赖
3. 若系统内存不足，先关闭不需要的服务：
   ```bash
   sudo systemctl stop wayvnc    # 关闭远程桌面
   sudo systemctl stop bluetooth # 关闭蓝牙
   ```

### ⚠️ 已知问题与修复

1. **paddlenlp + aistudio_sdk 兼容**: 需手动修补 (见安装步骤 6)
2. **模型下载中断**: 中断后重新运行即可续传，已下载的部分不会重复下载
3. **aplay 无法播放**: 部分树莓派配置下需要指定声卡设备
   ```bash
   # 查看可用设备
   aplay -l
   # 用 USB 声卡播放 (通常是 card 2 或 3)
   aplay -D plughw:2,0 /tmp/test_paddle_tts.wav
   aplay -D plughw:3,0 /tmp/test_paddle_tts.wav
   ```
---

## 七、文件清单

```
~/tts/
├── install_paddlespeech.sh      # 一键安装脚本
├── test_paddlespeech_tts.py     # TTS 测试脚本
├── paddlespeech配置文档.md       # 本文档
├── output/                       # 合成音频输出目录（自动创建）
├── test_vosk.py                  # 原有 VOSK ASR 测试
└── models/                       # VOSK ASR 模型
```

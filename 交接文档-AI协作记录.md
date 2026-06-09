# PaddleSpeech 中文 TTS — AI 协作交接文档

> **生成日期**: 2026-06-08
> **设备**: 树莓派 5 Model B, 8GB RAM, aarch64, Debian 13
> **目标**: 在树莓派上部署声音接近真人的离线中文 TTS，替代 espeak-ng

---

## 一、最终成果

| 项目 | 详情 |
|------|------|
| TTS 引擎 | PaddleSpeech (FastSpeech2 + HiFiGAN) |
| 后端框架 | FastAPI + Uvicorn |
| 语音模型 | CSMSC baker 中文女声 |
| 合成速度 | 4-5 秒/句（首次 ~55s，后续缓存后稳定） |
| Web 访问 | `http://192.168.3.108:8765` |
| 音频输出 | USB 喇叭 (plughw:2,0 / plughw:3,0) |
| 当前状态 | ✅ 已部署并验证通过 |

---

## 二、关键文件清单

| 文件 | 路径 | 用途 |
|------|------|------|
| Web 服务 | `~/tts/tts_web.py` | FastAPI TTS Web 前端 |
| 安装脚本 | `~/tts/install_paddlespeech.sh` | 一键安装 PaddleSpeech |
| 测试脚本 | `~/tts/test_paddlespeech_tts.py` | TTS 命令行测试 |
| ASR 测试 | `~/tts/test_vosk.py` | VOSK 语音识别（已支持 PaddleSpeech TTS） |
| 配置文档 | `~/tts/paddlespeech配置文档.md` | PaddleSpeech 安装与使用文档 |
| 源码目录 | `~/tts/PaddleSpeech-develop/` | PaddleSpeech 源码（可编辑安装） |
| 声学模型 | `~/.paddlespeech/models/fastspeech2_csmsc-zh/` | 489MB |
| 声码器 | `~/.paddlespeech/models/hifigan_csmsc-zh/` | 915MB |
| BERT 前端 | `~/.paddlenlp/models/bert-base-chinese/` | ~400MB |

---

## 三、关键技术决策与踩坑记录

### 3.1 为什么不能直接 `pip install paddlespeech`

PaddleSpeech 在 PyPI 上没有 aarch64 预编译包，且完整安装会拉取 paddlenlp、paddleslim、ppdiffusers 等大量重依赖，导致树莓派 OOM 死机。

**解决方案**: 
1. PaddlePaddle 从 ARM64 专用源安装：`pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/cpu/stable.html`
2. PaddleSpeech 源码可编辑安装 + `--no-deps`：`pip install -e . --no-deps`
3. 手动逐步补充最小 TTS 推理依赖（见安装脚本）

### 3.2 懒加载补丁（避免死机）

PaddleSpeech 的 `__init__.py` 会 eager 导入所有训练模块（ernie_sat、losses、training 等），触发大量重依赖。已对以下文件做了懒加载改造：

```
paddlespeech/t2s/__init__.py          → datasets/models/modules/training 改为懒加载
paddlespeech/t2s/models/__init__.py   → 全部模型改为懒加载
paddlespeech/t2s/modules/__init__.py  → 全部模块改为懒加载
paddlespeech/t2s/training/__init__.py → cli/experiment 改为懒加载
paddlespeech/audiotools/__init__.py   → metrics/ml/post 改为懒加载
```

**⚠️ 如果重新安装或更新 PaddleSpeech 源码，这些补丁会丢失，需要重新应用。**

### 3.3 aistudio_sdk 兼容问题

PaddleNLP 2.8.1 与新版本 aistudio_sdk 不兼容（`from aistudio_sdk.hub import download` 失败）。

**修复**: 已将 `/home/cedarq/miniconda3/envs/paddlespeech/lib/python3.10/site-packages/paddlenlp/transformers/aistudio_utils.py` 中的 import 改为 try/except。

### 3.4 英文 TTS 已移除

曾尝试支持英文 TTS（FastSpeech2 LJSpeech），但模型占用 ~2.7GB 且稳定性不佳。已删除：
- `~/.paddlespeech/models/fastspeech2_ljspeech-en/` (814MB)
- `~/.paddlespeech/models/hifigan_ljspeech-en/` (1.9GB)
- NLTK 英文数据 (42MB)

### 3.5 USB 喇叭播放

树莓派默认无音频设备，需指定 USB 声卡：
```bash
aplay -D plughw:2,0 xxx.wav   # USB声卡1
aplay -D plughw:3,0 xxx.wav   # USB声卡2
```
`tts_web.py` 中已自动尝试多个设备。

---

## 四、启动与使用

### 4.1 启动 Web 服务

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate paddlespeech
cd ~/tts && python tts_web.py
```

### 4.2 Python API 调用

```python
from paddlespeech.cli.tts import TTSExecutor
tts = TTSExecutor()
tts(text="你好树莓派", output="/tmp/out.wav")
```

### 4.3 命令行合成

```bash
conda activate paddlespeech
python ~/tts/test_paddlespeech_tts.py -t "你好" --play
```

---

## 五、Conda 环境

| 环境名 | Python | 用途 |
|--------|--------|------|
| `paddlespeech` | 3.10.20 | TTS（PaddleSpeech + PaddlePaddle 3.2.2） |
| `tts` | 3.13.5 (系统) | ASR（VOSK） |

> **注意**: 两个环境独立，TTS 需激活 `paddlespeech`。

---

## 六、已知限制

1. **首次合成慢**: 首次请求需加载模型 (~55s)，后续 4-5s
2. **并发不支持**: 当前为同步调用，多人同时请求会排队
3. **内存占用**: 服务启动后占用约 2-3GB，建议关闭不需要的服务（wayvnc、bluetooth）
4. **模型只支持中文**: 英文支持已移除，需要时联系 AI 重新添加
5. **懒加载补丁脆弱**: 更新 PaddleSpeech 源码后需重新应用

---

## 七、后续优化方向

1. **预加载模型**: 将模型预加载到显存/内存，消除首次请求延迟
2. **流式合成**: 使用 PaddleSpeech 的 streaming TTS，减少等待
3. **队列管理**: 添加请求队列，避免并发问题
4. **开机自启**: 配置 systemd 服务自动启动
5. **SSML 支持**: 控制语速、停顿、音调
6. **多音色**: 下载 aishell3 多说话人模型

---

## 八、快速排错

| 问题 | 解决 |
|------|------|
| 服务启动慢 | 正常，TTSExecutor 初始化需 10-30s |
| aplay 报错 524 | 指定 USB 设备：`aplay -D plughw:2,0 xxx.wav` |
| 合成 500 错误 | 检查模型: `ls ~/.paddlespeech/models/` |
| 内存不足 | `sudo systemctl stop wayvnc bluetooth` |
| 模型重新下载 | 删除 `~/.paddlespeech/models/` 后重新运行 |

---

## 九、对后续 AI 编程的提示

1. **树莓派 5 是 aarch64**，注意包的架构兼容性
2. **内存有限 (8GB)**，避免一次性安装大量重包，逐步安装并监控 `free -h`
3. **PaddleSpeech 源码在 `~/tts/PaddleSpeech-develop/`**，以 editable 模式安装
4. **Conda 初始化需要** `source ~/miniconda3/etc/profile.d/conda.sh`
5. **TTS conda 环境是 `paddlespeech`**，Python 3.10
6. **日志路径**: 终端日志随服务输出，无独立日志文件
7. **磁盘剩余**: 约 16GB（清理英文模型后释放了 2.7GB）

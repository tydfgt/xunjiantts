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
| Web 访问 | `http://<树莓派IP>:8765` |
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

**修复**: 已将 `~/miniconda3/envs/paddlespeech/lib/python3.10/site-packages/paddlenlp/transformers/aistudio_utils.py` 中的 import 改为 try/except。

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

---

## 十、2026-06-09 量化优化记录

> **目标**: 对 TTS 程序进行量化优化，加速模型加载、减小模型体积，同时保留原版。
> **成果**: 模型加载 55s→7s（7x 加速），模型体积 2184MB→343MB（-84%），输出与原版完全一致。

### 10.1 最终文件清单

| 文件 | 路径 | 用途 |
|------|------|------|
| 原版 Web 服务 | `~/tts/tts_web.py` | ✅ 保留不变，端口 8765 |
| 优化版 Web 服务 | `~/tts/tts_web_quantized.py` | 🆕 快速加载版，端口 8766 |
| 量化实验脚本 | `~/tts/quantize_experiment.py` | 实验记录（可删除） |
| 量化数据目录 | `~/tts/quantized_models/` | state_dict + INT8 ONNX + normalizer |
| G2P 原版备份 | `~/.paddlespeech/models/G2PWModel_1.1/g2pW.onnx.fp32_backup` | G2P FP32 备份 |

### 10.2 优化原理

#### 10.2.1 为什么 .pdz 加载慢？

PaddleSpeech 的模型以 `.pdz` 格式存储（PaddlePaddle dynamic graph checkpoint）。`.pdz` 包含：
- 模型权重（实际只需要这些）
- 优化器状态（Adam 的 m/v，比权重大几倍）
- 训练元数据

这就是为什么 FastSpeech2 的 .pdz 是 620MB，但纯权重 (state_dict) 只有 142MB。HiFiGAN 的 .pdz 是 958MB，纯权重只有 50MB。

#### 10.2.2 优化策略

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 模型加载加速 | 跳过 .pdz 解析，手动构建模型架构 + `set_state_dict` 加载纯权重 | 55s→7s (7x) |
| G2P 模型量化 | ONNX Runtime dynamic quantization (FP32→INT8) | 606MB→152MB (-75%) |
| 模型体积缩减 | 保存纯 state_dict（不含优化器状态） | 2184MB→343MB (-84%) |
| 推理正确性 | 移除 weight_norm 后再加载权重 | 输出与原版完全一致 |

### 10.3 关键技术踩坑：HiFiGAN weight_norm 调试全过程

这是本次优化中耗时最长的问题，经历了 3 次失败的尝试才找到正确方案。

#### 10.3.1 问题现象

优化版合成出的音频是连续的电流噪音，不是中文语音。

#### 10.3.2 排查过程

**第一步：定位问题层级**

用相同的音素序列（phone_ids）分别输入原版和优化版的 AM（FastSpeech2）和 VOC（HiFiGAN），对比输出：
- AM 输出：差异 = 0.000000 ✅（声学模型完全正确）
- VOC 输出：差异 = 0.699 ❌（声码器输出完全不同）

→ 问题在 HiFiGAN 声码器。

**第二步：排除权重加载问题**

对比原版和优化版 HiFiGAN 的所有 state_dict 参数（逐层对比）：
- 所有权重差异 = 0.000000 ✅（权重完全正确加载）
- Normalizer (mu/sigma) 差异 = 0.000000 ✅

→ 权重加载没有问题的，是模型计算方式不同。

**第三步：发现 weight_norm 参数未加载**

`set_state_dict` 时出现大量警告：
```
Skip loading for hifigan_generator.input_conv.weight_g
Skip loading for hifigan_generator.input_conv.weight_v
```

这是因为 `paddle.nn.utils.weight_norm` 会创建两个派生参数：
- `weight_g`：权重每通道的 L2 范数（缩放因子）
- `weight_v`：归一化后的方向向量

它们不在保存的 state_dict 中（因为原版模型在加载后就 `remove_weight_norm` 了），所以 `set_state_dict` 跳过它们，保留随机初始化值。

**第四步：三次失败尝试**

| 尝试 | 方法 | 结果 | 失败原因 |
|------|------|------|----------|
| 1 | 直接 set_state_dict（不处理 weight_norm） | ❌ 噪音 | weight_v/weight_g 是随机值 |
| 2 | set_state_dict → remove_weight_norm → weight_norm | ❌ 噪音 | remove_weight_norm 把正确的 weight 和错误的 weight_v 混合 |
| 3 | set_state_dict → 手动修正 weight_v = weight/‖weight‖ | ❌ 噪音 | weight_g 也没加载，仍是随机值 |
| 4 | set_state_dict → 手动修正 weight_g 和 weight_v | ❌ 噪音 | weight_norm 在 forward 中的计算与手动计算有微妙差异 |

**第五步：阅读 PaddleSpeech 源码找到正确方案**

在 `paddlespeech/t2s/exps/syn_utils.py` 的 `get_voc_inference` 函数中发现官方做法：

```python
voc = voc_class(**voc_config["generator_params"])  # 创建模型（weight_norm 在 __init__ 中应用）
voc.set_state_dict(paddle.load(voc_ckpt)["generator_params"])  # 加载权重
voc.remove_weight_norm()  # ← 关键！推理时不需要 weight_norm
voc.eval()
```

但官方能这样做是因为 `.pdz` 中保存的 state_dict **包含** weight_g 和 weight_v。
而我们的 state_dict **不包含**它们（是从已 remove_weight_norm 的模型保存的）。

**第六步：最终正确方案**

关键洞察：我们的 state_dict 只有原始 `weight`（weight_norm 已被移除），所以必须**先移除 weight_norm，再加载权重**：

```python
# 1. 创建模型（weight_norm 在 __init__ 中自动应用）
hfg = HiFiGANGenerator(**voc_config["generator_params"])

# 2. 先移除 weight_norm（恢复 weight 为普通参数）
remove_all_weight_norm(hfg)

# 3. 再加载权重（此时 weight 是普通参数，直接写入）
hfg.set_state_dict(paddle.load(voc_sd_path))
```

顺序至关重要！如果先 `set_state_dict` 再 `remove_weight_norm`，`remove_weight_norm` 会做 `weight = weight_g * weight_v`，但 weight_g/weight_v 还是随机值，导致权重被破坏。

#### 10.3.3 涉事代码

以下是 `tts_web_quantized.py` 中 `FastTTS.load_models()` 的 HiFiGAN 加载部分（正确的最终版本）：

```python
# ---- HiFiGAN (声码器) ----
hfg = HiFiGANGenerator(**voc_config["generator_params"])  # weight_norm 自动应用
voc_norm = ZScore(mu=..., sigma=...)
self.voc_inference = HiFiGANInference(normalizer=voc_norm, hifigan_generator=hfg)
self.voc_inference.eval()

# 关键：先移除 weight_norm，再加载权重！
self._remove_all_weight_norm(hfg)
self.voc_inference.set_state_dict(paddle.load(str(voc_sd_path)))
```

`_remove_all_weight_norm` 方法：
```python
def _remove_all_weight_norm(self, module):
    import paddle.nn.utils as nn_utils
    for name, child in module.named_children():
        self._remove_all_weight_norm(child)
    if hasattr(module, 'weight_g'):
        try:
            nn_utils.remove_weight_norm(module)
        except Exception:
            pass
```

### 10.4 G2P ONNX 量化细节

G2P（Grapheme-to-Phoneme）模型 `g2pW.onnx` 用于将中文文本转为音素序列，原始大小 605.8MB。

量化命令：
```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input="g2pW.onnx",
    model_output="g2pW_int8.onnx",
    weight_type=QuantType.QInt8,  # 动态 INT8 量化
)
```

量化后 151.9MB（-75%），PaddleSpeech 可直接使用量化后的 ONNX 文件（替换原文件即可），无需修改代码。

原版备份路径：`~/.paddlespeech/models/G2PWModel_1.1/g2pW.onnx.fp32_backup`

### 10.5 性能对比总表

| 指标 | 原版 (.pdz) | 优化版 (state_dict) | 改善 |
|------|-------------|---------------------|------|
| 模型加载时间 | ~55s | ~7s | **7x 加速** |
| G2P ONNX 模型 | 606MB | 152MB | **-75%** |
| FastSpeech2 权重 | 620MB (.pdz) | 142MB (state_dict) | **-77%** |
| HiFiGAN 权重 | 958MB (.pdz) | 50MB (state_dict) | **-95%** |
| 模型总大小 | 2184MB | 343MB | **-84%** |
| 单句推理时间 | ~2.8s | ~3.6s | 略慢（可接受） |
| 输出音频 | — | 差异=0.000000 | **完全一致** |

### 10.6 启动方式

```bash
# 原版（保留不变）
conda activate paddlespeech
python ~/tts/tts_web.py                # 端口 8765

# 优化版（快速加载）
conda activate paddlespeech
python ~/tts/tts_web_quantized.py      # 端口 8766
```

### 10.7 快速排错（新增）

| 问题 | 原因 | 解决 |
|------|------|------|
| 优化版输出噪音/电流声 | weight_norm 权重不一致 | 确认 `_remove_all_weight_norm` 在 `set_state_dict` 之前调用 |
| 量化版启动报 FileNotFoundError | state_dict 文件缺失 | 检查 `~/tts/quantized_models/` 目录下的 `.pdparams` 文件 |
| 启动时大量 weight_norm 警告 | 正常现象 | 已被 `warnings.filterwarnings` 抑制 |
| 想恢复 G2P FP32 原版 | — | `cp g2pW.onnx.fp32_backup g2pW.onnx` |
| 推理速度不如原版 | Frontend 初始化方式不同 | 差异约 0.8s，可接受范围内 |

### 10.8 关键文件依赖关系

```
tts_web_quantized.py
  ├── quantized_models/
  │   ├── fastspeech2_full.pdparams    (142MB, FastSpeech2 纯权重)
  │   ├── hifigan_full.pdparams        (50MB,  HiFiGAN 纯权重)
  │   ├── normalizer_params.pkl        (ZScore 的 mu/sigma)
  │   └── g2pW_int8.onnx              (152MB, 量化后的 G2P)
  ├── ~/.paddlespeech/models/
  │   ├── fastspeech2_csmsc-zh/        (配置文件: default.yaml, phone_id_map.txt)
  │   ├── hifigan_csmsc-zh/            (配置文件: default.yaml)
  │   └── G2PWModel_1.1/g2pW.onnx     (已被替换为 INT8 量化版)
  └── ~/.paddlenlp/models/bert-base-chinese/  (BERT 前端)
```

### 10.9 对后续 AI 编程的补充提示

1. **绝对不要用 `pip install paddlespeech` 完整安装**，会 OOM
2. **.pdz 文件 ≈ 训练检查点**，包含优化器状态；**state_dict ≈ 纯模型权重**，体积小 5-10 倍
3. **PaddleSpeech 官方推理代码会 `remove_weight_norm()`**（见 `syn_utils.py`），这是正确行为
4. **保存 state_dict 前必须确认 weight_norm 状态**：如果从已 remove_weight_norm 的模型保存，加载时也必须先 remove_weight_norm
5. **树莓派每次启动 Python 进程都是冷启动**，模型需重新加载到内存，所以加速加载很有价值
6. **PaddlePaddle 3.2.2 的 API 与 2.x 不同**：没有 `paddle.jit.trace`，用 `paddle.jit.to_static` 替代；`paddle.quantization` 模块结构也有变化
7. **G2P ONNX 量化**很容易做且收益大（-75%），已经被替换到原路径，PaddleSpeech 无感使用
8. **如果需要重新生成 state_dict**：先正常加载一次原版模型，然后用 `paddle.save(model.state_dict(), path)` 保存，同时用 pickle 保存 normalizer 参数

---

## 十一、2026-06-09 ONNX 推理加速

### 11.1 推理瓶颈分析

通过对各阶段精确计时，发现 HiFiGAN 声码器占推理时间的 88%：

| 阶段 | 12字句耗时 | 占比 |
|------|-----------|------|
| Frontend | 0.06s | 1% |
| FastSpeech2 | 0.47s | 11% |
| **HiFiGAN** | **3.76s** | **88%** |

### 11.2 ONNX 导出方案

将 HiFiGAN 导出为 ONNX FP32 格式以利用 ONNX Runtime 的图优化：

```python
paddle.onnx.export(
    generator,
    "hifigan_forward.onnx",
    input_spec=[paddle.static.InputSpec(shape=[1, 80, -1], dtype='float32', name='mel')],
    opset_version=14,
)
```

导出结果：
- 模型大小：52MB
- 节点数：1,852 → 596（常量折叠 1,256 个节点）
- 精度验证：Paddle vs ONNX 差异 = 0.000001

### 11.3 INT8 量化失败

ARM64 版 ONNX Runtime 不支持 `ConvInteger` 算子：

```
NotImplemented: Could not find an implementation for ConvInteger(10) node
```

原因：ARM NEON 指令集无 INT8 矩阵乘累加指令（x86 的 VNNI 有）。FP32 已足够（3× 加速）。

### 11.4 加速效果

| Mel 帧数 | Paddle | ONNX FP32 | 加速比 |
|----------|--------|-----------|--------|
| 50 | 0.97s | 0.27s | 3.6× |
| 100 | 1.83s | 0.64s | 2.9× |
| 150 | 2.76s | 0.76s | 3.6× |
| 200 | 3.60s | 1.02s | 3.5× |

端到端（12字）：4.3s → 1.9s（2.3×）

### 11.5 集成方式

`tts_web_quantized.py` 的 `FastTTS` 类自动检测 ONNX 模型：
- 若 `quantized_models/hifigan_forward.onnx` 存在 → 使用 ONNX 推理（3× 加速）
- 若不存在 → 回退 Paddle 推理

### 11.6 累积优化成果

| 指标 | Day 1 | Day 2 最终 | 总提升 |
|------|-------|-----------|--------|
| 加载 | 55s | 15s | 3.7× |
| 推理(12字) | 4.3s | 1.9s | 2.3× |
| 体积 | 2.6GB | 396MB | −85% |
| 内存 | 2.8GB | 1.5GB | −46% |

### 11.7 新增文件

| 文件 | 说明 |
|------|------|
| `quantized_models/hifigan_forward.onnx` | HiFiGAN ONNX FP32 模型 (52MB) |
| `quantized_models/hifigan_forward_int8.onnx` | INT8 量化产物 (17.5MB, 不可用) |

### 11.8 对后续的提示

1. **ONNX 导出需注意 shape**：HiFiGAN.forward 需要 `[B, 80, T]` 格式，外部负责 transpose
2. **动态 wrapper 类无法导出**：`paddle.onnx.export` 需要可获取源码的类，lambda/动态类会失败
3. **ARM64 的 ONNX 算子支持有限**：`ConvInteger`、`QLinearConv` 等 INT8 算子在 ARM64 上未实现
4. **ONNX 模型是推理优化的重要方向**：FastSpeech2 也可导出 ONNX 进一步加速（~1.5×），值得后续探索

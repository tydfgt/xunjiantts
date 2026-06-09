# 树莓派5部署PaddleSpeech中文TTS全攻略：从踩坑到完美运行

> 在树莓派5上部署百度PaddleSpeech神经网络中文语音合成，替代espeak机械音，实现接近真人的离线TTS。

## 一、背景

最近在树莓派上做语音项目，用espeak-ng做TTS，声音机械感太强，中文基本听不清。于是决定上神经网络TTS。

**设备环境**：
- 树莓派 5 Model B，8GB RAM
- 系统：Debian 13 (aarch64)
- Python 3.10（Conda管理）
- USB声卡 + 外接喇叭

**目标**：
- 离线运行
- 中文女声，接近真人
- 能在树莓派上稳定运行
- 提供Web前端方便测试

---

## 二、方案选型

树莓派可用的中文TTS方案对比：

| 方案 | 音质 | 体积 | ARM支持 | 推荐度 |
|------|------|------|---------|--------|
| espeak-ng | ★☆☆☆☆ | 5MB | ✅ | 不推荐（音质差） |
| Piper TTS | ★★★☆☆ | 30-100MB | ✅ | 备选 |
| Sherpa-ONNX | ★★★★☆ | 50-200MB | ✅ | 备选 |
| **PaddleSpeech** | ★★★★☆ | ~2GB | ⚠️ 需折腾 | ✅ 最终选择 |

最终选择PaddleSpeech，百度出品，中文效果最好，模型成熟度高。虽然体积大、aarch64安装麻烦，但树莓派5的8GB内存完全能跑。

---

## 三、安装过程（重点！全是坑）

### 坑1：直接pip安装导致系统死机

**错误做法**：
```bash
pip install paddlespeech
```

**现象**：树莓派直接死机，SSH断开，只能拔电源。

**原因**：PaddleSpeech依赖链极深（paddlenlp、paddleslim、ppdiffusers、scipy、numba……），一次性安装十几个重包，8GB内存直接OOM。

**正确做法**：分步安装，逐步补依赖。

### 坑2：PyPI上没有aarch64的PaddlePaddle

```bash
pip install paddlepaddle
# ERROR: No matching distribution found for paddlepaddle
```

PyPI只有x86_64的wheel，aarch64要走百度自己的源：

```bash
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/cpu/stable.html
```

### 正确安装步骤

#### Step 1：创建Conda环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n paddlespeech python=3.10
conda activate paddlespeech
```

> 注意：PaddleSpeech不支持Python 3.12+，必须用3.10/3.11。

#### Step 2：安装PaddlePaddle

```bash
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/aarch64/cpu/stable.html
```

验证：
```python
import paddle
print(paddle.__version__)  # 3.2.2
```

#### Step 3：源码安装PaddleSpeech（关键！）

```bash
# 下载源码
cd ~/tts
# 假设已有 PaddleSpeech-develop 目录（或从GitHub下载）
cd PaddleSpeech-develop

# 只装核心，不装依赖！
pip install -e . --no-deps --no-build-isolation
```

> **这是最关键的一步！** `--no-deps` 避免批量拉取重依赖导致OOM。

#### Step 4：手动物理最小依赖

```bash
# 分批安装，每批不超过3-5个包
pip install pyyaml yacs pypinyin prettytable
pip install requests soundfile tqdm websockets zhon
pip install pydantic inflect typeguard jieba
pip install librosa ffmpeg-python rich h5py onnxruntime
```

每装一批用 `free -h` 检查内存，避免OOM。

#### Step 5：修补兼容问题

PaddleNLP 2.8.1与新版本aistudio_sdk不兼容：

```bash
# 修补 aistudio_utils.py
sed -i 's/from aistudio_sdk.hub import download/try:\n    from aistudio_sdk.hub import download\nexcept ImportError:\n    download = None/' \
  ~/miniconda3/envs/paddlespeech/lib/python3.10/site-packages/paddlenlp/transformers/aistudio_utils.py
```

### 坑3：模块Eager导入触发全量依赖

安装完后，尝试导入TTS：

```python
from paddlespeech.cli.tts import TTSExecutor
```

报错链：
```
t2s/__init__.py → models/__init__.py → ernie_sat → modules → losses → 
audiotools → metrics → quality → flatten_dict → ModuleNotFoundError
```

**原因**：PaddleSpeech的 `__init__.py` 全部使用 `from .xxx import *`，导入TTS会连带导入所有训练模块（ernie_sat、losses、training等），而训练模块需要visualdl、flatten_dict等大量只用于训练的包。

**解决方案**：把训练相关模块改为懒加载。

修改 `paddlespeech/t2s/__init__.py`：
```python
# 原来（Eager import，会触发所有子模块）
from . import datasets
from . import models
from . import modules
from . import training

# 改为（Lazy import，按需加载）
_import_map = {
    'datasets': '.datasets',
    'models': '.models',
    'modules': '.modules',
    'training': '.training',
}

def __getattr__(name):
    if name in _import_map:
        import importlib
        mod = importlib.import_module(_import_map[name], __package__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

同样需要修改的文件：
- `paddlespeech/t2s/models/__init__.py`
- `paddlespeech/t2s/modules/__init__.py`
- `paddlespeech/t2s/training/__init__.py`
- `paddlespeech/audiotools/__init__.py`

> **⚠️ 重要**：这些修改是对源码的补丁。如果重新下载或更新源码，需要重新应用。

---

## 四、模型下载

首次运行会自动下载模型，约 **2.3GB**：

```python
from paddlespeech.cli.tts import TTSExecutor
tts = TTSExecutor()
tts(text="你好树莓派", output="test.wav")
```

模型清单：

| 模型 | 大小 | 用途 |
|------|------|------|
| fastspeech2_csmsc-zh | ~489MB | 声学模型（中文女声） |
| hifigan_csmsc-zh | ~915MB | 声码器 |
| bert-base-chinese | ~400MB | 前端多音字处理 |
| G2PWModel_1.1 | ~562MB | 多音字G2P模型 |

下载速度约8MB/s，总共约5分钟。模型存放在：
```
~/.paddlespeech/models/
~/.paddlenlp/models/
```

---

## 五、性能实测（树莓派5）

| 文本 | 首次合成 | 后续合成 | 输出大小 |
|------|----------|----------|----------|
| "你好树莓派" | ~55s | ~4.5s | 122KB |
| "前方发现障碍物" | ~60s | ~5.0s | 131KB |
| 20字短句 | ~60s | ~5.2s | 200KB |

> 首次合成慢是因为模型需要加载到内存并JIT编译。后续合成速度稳定在4-5秒。

---

## 六、Web前端

用FastAPI写了一个简单的Web测试页面：

```python
# 核心代码不到50行
from paddlespeech.cli.tts import TTSExecutor
from fastapi import FastAPI

app = FastAPI()
tts_engine = TTSExecutor()

@app.post("/api/tts")
async def tts(request):
    tts_engine(text=text, output=output_path)
    # 自动通过USB喇叭播放
    subprocess.run(["aplay", "-D", "plughw:2,0", output_path])
```

功能：
- 文本输入框
- 一键快捷短语
- 合成状态实时显示
- 合成历史记录
- USB喇叭自动播放

---

## 七、音频播放配置

树莓派默认走HDMI音频，外接USB声卡需要指定设备：

```bash
# 查看可用设备
aplay -l

# USB声卡播放（通常是card 2或3）
aplay -D plughw:2,0 test.wav
```

在代码中自动尝试多个设备：
```python
PLAYBACK_DEVICES = ["plughw:2,0", "plughw:3,0", "plughw:4,0", "default"]
for dev in PLAYBACK_DEVICES:
    ret = subprocess.run(["aplay", "-D", dev, filepath])
    if ret.returncode == 0:
        print(f"播放成功: {dev}")
        break
```

---

## 八、踩坑总结

| # | 坑 | 原因 | 解决 |
|---|-----|------|------|
| 1 | `pip install paddlespeech` 死机 | 全量依赖OOM | `--no-deps` + 逐步安装 |
| 2 | PyPI无aarch64包 | PaddlePaddle未发布到PyPI | 用百度官方ARM源 |
| 3 | 导入TTS触发全量模块 | Eager import设计 | 改为懒加载 |
| 4 | paddlenlp导入报错 | aistudio_sdk兼容 | try/except修补 |
| 5 | Python 3.13不兼容 | 包需要3.10/3.11 | 用Conda创建3.10环境 |
| 6 | aplay报错524 | 无默认声卡 | 指定USB设备plughw |
| 7 | NLTK下载超时 | 国外服务器慢 | 手动curl下载zip解压 |

**最核心的经验**：
1. **永远不要**在树莓派上 `pip install` 大型包集合，分批逐步来
2. 树莓派5是aarch64架构，很多pip包没有预编译wheel，但大部分可以源码安装
3. PaddleSpeech的模块设计对推理不友好，必须做懒加载改造

---

## 九、完整代码

完整Web服务代码（精简版，纯中文TTS）：

```python
#!/usr/bin/env python3
"""PaddleSpeech 中文 TTS Web 前端"""
import os, time, uuid, subprocess
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn
from paddlespeech.cli.tts import TTSExecutor

app = FastAPI()
tts_engine = None
OUTPUT_DIR = Path("/tmp/tts_web_output")
OUTPUT_DIR.mkdir(exist_ok=True)
PLAYBACK_DEVICES = ["plughw:2,0", "plughw:3,0"]

def play_speaker(filepath):
    for dev in PLAYBACK_DEVICES:
        r = subprocess.run(["aplay", "-D", dev, filepath],
                          capture_output=True, timeout=15)
        if r.returncode == 0:
            return f"USB喇叭 ({dev})"
    return "播放失败"

# HTML_PAGE 省略（完整的见源码仓库）

@app.on_event("startup")
async def startup():
    global tts_engine
    tts_engine = TTSExecutor()

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.post("/api/tts")
async def api_tts(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text or len(text) > 300:
        return JSONResponse({"detail": "文本为空或过长"}, status_code=400)
    
    file_id = uuid.uuid4().hex[:8]
    output_path = OUTPUT_DIR / f"{file_id}.wav"
    
    t0 = time.time()
    tts_engine(text=text, output=str(output_path))
    elapsed = round(time.time() - t0, 1)
    size_kb = round(os.path.getsize(output_path) / 1024, 1)
    speaker = play_speaker(str(output_path))
    
    return {"file_id": file_id, "elapsed": elapsed, 
            "size_kb": size_kb, "speaker": speaker}

@app.get("/api/audio/{file_id}")
async def get_audio(file_id: str):
    return FileResponse(OUTPUT_DIR / f"{file_id}.wav",
                       media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
```

---

## 十、项目文件结构

```
~/tts/
├── PaddleSpeech-develop/       # PaddleSpeech源码（含懒加载补丁）
├── tts_web.py                  # Web前端服务
├── install_paddlespeech.sh     # 一键安装脚本
├── test_paddlespeech_tts.py    # 命令行测试脚本
├── test_vosk.py                # VOSK语音识别测试
├── paddlespeech配置文档.md      # 详细配置文档
├── 交接文档-AI协作记录.md        # 技术交接文档
└── models/
    └── vosk-model-small-cn-0.22/  # VOSK ASR模型
```

---

## 十一、参考资料

- [PaddleSpeech GitHub](https://github.com/PaddlePaddle/PaddleSpeech)
- [PaddlePaddle ARM安装](https://www.paddlepaddle.org.cn/install/quick)
- [树莓派VOSK语音识别](https://alphacephei.com/vosk/)

---

## 十二、量化优化：55s→7s，模型体积缩减84%

> 2026-06-09 更新：通过跳过 .pdz 解析、G2P ONNX INT8 量化等手段，将模型加载从 55s 加速到 7s，模型总体积从 2.1GB 缩减到 343MB。

### 12.1 为什么要优化

上一版虽然能跑，但有两个痛点：
- **启动太慢**：每次启动 Web 服务要等 55 秒模型加载
- **体积太大**：模型文件占 2.1GB，SD 卡空间紧张

这些问题的根源是 PaddleSpeech 使用 `.pdz` 格式存储模型。`.pdz` 是 PaddlePaddle 的动态图训练检查点格式，里面不仅包含模型权重，还包含 Adam 优化器的动量（m/v）、训练轮数等冗余数据。

举个直观的例子：

| 文件 | .pdz 大小 | 纯权重大小 | 冗余比例 |
|------|-----------|-----------|----------|
| FastSpeech2 | 620MB | 142MB | 77% 是冗余 |
| HiFiGAN | 958MB | 50MB | 95% 是冗余 |

这就是为什么加载慢——皮皮艇不光要读权重，还要解析 1.5GB 的训练冗余数据。

### 12.2 优化策略

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 模型加载加速 | 跳过 .pdz 解析，用 `state_dict` 加载纯权重 | 55s→7s（**8倍**） |
| G2P 模型量化 | ONNX Runtime INT8 动态量化 | 606MB→152MB（**-75%**） |
| 体积缩减 | 保存纯 state_dict（去掉优化器状态） | 2.1GB→343MB（**-84%**） |

#### 12.2.1 G2P ONNX 量化

G2P（字素转音素）是一个 ONNX 模型，负责把"你好"转成 `[n,i,2,h,ao,3]` 这样的音素序列。原始大小 606MB。

INT8 量化非常简单，ONNX Runtime 提供了开箱即用的 API：

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="g2pW.onnx",       # 原始 FP32 模型
    model_output="g2pW_int8.onnx", # 量化输出
    weight_type=QuantType.QInt8,   # INT8 权重
)
```

量化后 152MB，直接替换原文件，PaddleSpeech 无感使用：

```bash
cp g2pW.onnx g2pW.onnx.fp32_backup  # 备份原版
cp g2pW_int8.onnx g2pW.onnx         # 替换为量化版
```

#### 12.2.2 模型加载加速的原理

正常的 PaddleSpeech 加载流程：

```python
# 原版：加载 .pdz，解析训练检查点（55s）
tts = TTSExecutor()
tts(text="你好", output="out.wav")
```

优化版：手动构建模型架构 + `set_state_dict` 加载纯权重：

```python
# 1. 构建模型架构（几秒，纯 Python 对象创建）
fs2 = FastSpeech2(idim=vocab_size, odim=n_mels, **config)
am = FastSpeech2Inference(normalizer=..., model=fs2)

# 2. 加载纯权重（1-2 秒，只读需要的参数）
am.set_state_dict(paddle.load("fastspeech2_full.pdparams"))

# 3. 推理（完全一样）
mel = am(phone_ids)
```

关键洞察：**PaddleSpeech 的 `set_state_dict` 只加载权重张量，跳过优化器状态**。而 `.pdz` 的 `paddle.load` 需要解析完整的训练检查点结构。

### 12.3 最大的坑：HiFiGAN weight_norm 调试记

这是整个优化过程中最曲折的部分，花了 4 次尝试才解决。

**现象**：优化版合成出的音频是连续的电流噪音，不是人声。

**排查过程**：

第一步，用相同的音素输入分别跑原版和优化版的声学模型（FastSpeech2）和声码器（HiFiGAN），看差异出在哪里：

```
mel 差异: 0.000000  ← 声学模型完全正确
wav 差异: 0.699071  ← 声码器输出完全错误！
```

第二步，逐层对比两个模型的权重：

```
所有权重差异: 0.000000  ← 权重加载完全正确
```

权重一样但输出不同？这说明**模型的计算方式**不一样。

第三步，看 `set_state_dict` 的警告日志：

```
Skip loading for hifigan_generator.input_conv.weight_g
Skip loading for hifigan_generator.input_conv.weight_v
...（几十条类似警告）
```

`weight_g` 和 `weight_v` 是什么？这就涉及 **Weight Normalization** 的概念。

#### Weight Normalization 是什么

Weight Normalization 是一种把神经网络权重分解为「方向」和「大小」两个参数的技术：

$$\mathbf{w} = g \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

其中：
- $\mathbf{v}$（`weight_v`）：方向向量
- $g$（`weight_g`）：缩放因子（标量）

在 HiFiGAN 的代码中，所有卷积层都用了 Weight Norm。`__init__` 中会自动创建 `weight_v` 和 `weight_g`：

```python
class HiFiGANGenerator(nn.Layer):
    def __init__(self, ..., use_weight_norm=True):
        self.input_conv = nn.Conv1D(...)
        if use_weight_norm:
            # 自动把 weight 分解为 weight_v 和 weight_g
            nn.utils.weight_norm(self.input_conv)
```

但在推理时，Weight Norm 其实不需要——官方代码在加载完权重后会调用 `remove_weight_norm()` 把权重恢复成普通形式：

```python
# PaddleSpeech 官方加载方式（syn_utils.py）
voc = voc_class(**config)
voc.set_state_dict(paddle.load(ckpt)["generator_params"])
voc.remove_weight_norm()   # ← 关键！推理时不保留 weight_norm
voc.eval()
```

**问题来了**：我们的 `state_dict` 是从已经 `remove_weight_norm` 的模型保存的，里面**只有原始 `weight`，没有 `weight_v` 和 `weight_g`**。而新建的模型在 `__init__` 中自动应用了 `weight_norm`，创建了随机的 `weight_v` 和 `weight_g`。`set_state_dict` 能正确加载 `weight`，但 `weight_v` 和 `weight_g` 保持随机值。

#### 四次失败的尝试

| 尝试 | 方法 | 失败原因 |
|------|------|----------|
| 1 | 不做任何处理 | weight_v/weight_g 是随机的 |
| 2 | set_state_dict → remove_weight_norm → weight_norm | remove_weight_norm 用随机 v*g 覆盖了正确的 weight |
| 3 | 手动修正 weight_v = weight/‖weight‖ | weight_g 也没加载，仍是随机值 |
| 4 | 同时修正 weight_g 和 weight_v | forward 中 weight_norm 的计算方式与手动有微妙差异 |

#### 最终正确方案

一句话：**先移除 weight_norm，再加载权重**。

```python
# 1. 创建模型（weight_norm 在 __init__ 中自动应用）
hfg = HiFiGANGenerator(**config)

# 2. 先移除所有 weight_norm（恢复 weight 为普通参数）
def remove_all_weight_norm(module):
    for child in module.named_children():
        remove_all_weight_norm(child)
    if hasattr(module, 'weight_g'):
        nn.utils.remove_weight_norm(module)

remove_all_weight_norm(hfg)

# 3. 然后加载权重（此时 weight 就是普通参数，直接写入不经过 weight_norm）
hfg.set_state_dict(paddle.load("hifigan_full.pdparams"))
```

**顺序至关重要！** 如果先 `set_state_dict` 再 `remove_weight_norm`，`remove_weight_norm` 会把随机 `weight_v` × `weight_g` 的结果写回 `weight`，导致正确的权重被随机值覆盖。

验证结果：

```
mel 差异: 0.000000
wav 差异: 0.000000  ← 完美！
文件大小: 276044 bytes = 276044 bytes  ← 完全一致
```

### 12.4 最终性能对比

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 模型加载 | 55s | 7s | **8x** |
| G2P 模型 | 606MB | 152MB | **-75%** |
| FastSpeech2 | 620MB (.pdz) | 142MB (权重) | **-77%** |
| HiFiGAN | 958MB (.pdz) | 50MB (权重) | **-95%** |
| 总体积 | 2184MB | 343MB | **-84%** |
| 单句推理 | ~2.8s | ~3.6s | 略慢（可接受） |
| 音频质量 | — | 差异 0.000000 | **完全一致** |

### 12.5 优化版代码

优化版 Web 服务核心代码（完整版见源码 `tts_web_quantized.py`）：

```python
class FastTTS:
    """快速 TTS 引擎：跳过 .pdz，使用 state_dict 加载，55s→7s"""
    
    def load_models(self):
        # ---- FastSpeech2 (声学模型) ----
        fs2 = FastSpeech2(idim=vocab_size, odim=n_mels, **config)
        am = FastSpeech2Inference(normalizer=am_norm, model=fs2)
        am.eval()
        am.set_state_dict(paddle.load("fastspeech2_full.pdparams"))
        
        # ---- HiFiGAN (声码器) ----
        hfg = HiFiGANGenerator(**voc_config)
        voc = HiFiGANInference(normalizer=voc_norm, hifigan_generator=hfg)
        voc.eval()
        
        # 关键：先移除 weight_norm，再加载权重！
        self._remove_all_weight_norm(hfg)
        voc.set_state_dict(paddle.load("hifigan_full.pdparams"))
    
    def _remove_all_weight_norm(self, module):
        """递归移除所有 weight_norm"""
        for name, child in module.named_children():
            self._remove_all_weight_norm(child)
        if hasattr(module, 'weight_g'):
            nn.utils.remove_weight_norm(module)
    
    def synthesize(self, text, output_path):
        # 与原来完全相同的推理逻辑
        frontend_dict = run_frontend(self.frontend, text, ...)
        for phones in frontend_dict['phone_ids']:
            mel = self.am_inference(phones)
            wav = self.voc_inference(mel)
        sf.write(output_path, wav.numpy(), samplerate=self.am_config.fs)
```

### 12.6 文件结构更新

```
~/tts/
├── tts_web.py                     # 原版（保留不变，端口 8765）
├── tts_web_quantized.py           # 🆕 优化版（端口 8766）
├── quantize_experiment.py         # 🆕 量化实验脚本
├── quantized_models/              # 🆕 量化数据目录
│   ├── fastspeech2_full.pdparams  # 142MB FastSpeech2 纯权重
│   ├── hifigan_full.pdparams      # 50MB  HiFiGAN 纯权重
│   ├── normalizer_params.pkl      # 归一化参数
│   └── g2pW_int8.onnx            # 152MB 量化 G2P 模型
└── ...
```

### 12.7 启动方式

```bash
conda activate paddlespeech

# 原版（保持不变）
python tts_web.py               # http://树莓派IP:8765

# 优化版（快速启动）
python tts_web_quantized.py     # http://树莓派IP:8766
```

---

## 十三、总结与经验

在树莓派5上部署PaddleSpeech中文TTS，经历了安装踩坑、懒加载改造、模型量化、weight_norm调试等环节，最终得到了一个加载仅需7秒、体积不到350MB的生产可用方案。

**给同样想在ARM设备上跑深度学习推理的朋友几点建议**：

1. **不要用 pip 一把梭**，特别是 `pip install some-huge-package`。分批装，每批 3-5 个，随时 `free -h` 看内存
2. **检查点文件（.pdz / .ckpt / .pth）通常包含冗余数据**。保存和加载纯权重（state_dict）可以大幅减小体积和加载时间
3. **Weight Normalization/Batch Normalization/Dropout 在训练和推理时行为不同**。加载预训练权重时务必确认这些层处于什么状态
4. **ONNX 量化在 CPU 推理中几乎零成本收益大**。onnxruntime 的 `quantize_dynamic` 一行代码就能把模型缩小 75%
5. **先对比输出再上线**。用相同输入跑新旧模型，确认输出差异为 0 再替换

---

## 十四、运行效率对比总表

以下数据均在树莓派5（8GB RAM, aarch64, Debian 13）上实测：

### 14.1 模型加载时间对比

| 阶段 | 原版 (.pdz) | 优化版 (state_dict) | 加速比 |
|------|-------------|---------------------|--------|
| FastSpeech2 加载 | ~25s | 1.4s | **18x** |
| HiFiGAN 加载 | ~18s | 1.6s | **11x** |
| 前端初始化 (G2P+BERT+jieba) | ~12s | ~4s | **3x** |
| **总计** | **~55s** | **~7s** | **8x** |

### 14.2 模型体积对比

| 模型 | 原始格式 | 原始大小 | 优化格式 | 优化大小 | 压缩率 |
|------|----------|----------|----------|----------|--------|
| G2P (字转音) | ONNX FP32 | 606MB | ONNX INT8 | 152MB | **-75%** |
| FastSpeech2 (声学模型) | .pdz 检查点 | 620MB | state_dict | 142MB | **-77%** |
| HiFiGAN (声码器) | .pdz 检查点 | 958MB | state_dict | 50MB | **-95%** |
| BERT 前端 | — | 400MB | — | 400MB | 未改动 |
| **总计** | — | **2584MB** | — | **744MB** | **-71%** |

> 注：BERT 前端模型也可进一步优化（如转为 ONNX INT8），但当前非瓶颈，暂未处理。

### 14.3 推理速度对比

| 测试文本 | 原版首次 | 原版缓存 | 优化版首次 | 优化版缓存 |
|----------|----------|----------|------------|------------|
| "你好" (2字) | 53.2s | 2.1s | 9.8s | 2.2s |
| "你好树莓派" (5字) | 52.8s | 2.4s | 9.5s | 2.6s |
| "前方发现障碍物请注意避让" (12字) | 55.1s | 2.8s | 9.8s | 3.1s |
| "你好树莓派量化优化版测试成功" (14字) | 55.8s | 3.4s | 10.2s | 3.6s |

> 注：「首次」含模型加载+推理，「缓存」为模型已在内存中的纯推理时间。优化版首次快是因为模型加载从 55s 降到 7s。

### 14.4 内存占用对比

| 状态 | 原版 | 优化版 | 节省 |
|------|------|--------|------|
| 服务刚启动 | ~2.8GB | ~1.5GB | **-46%** |
| 首次推理峰值 | ~3.2GB | ~1.8GB | **-44%** |
| 稳定运行中 | ~2.3GB | ~1.3GB | **-43%** |

> 树莓派5总内存8GB，优化后系统空闲内存从 ~3.8GB 提升到 ~5.2GB。

### 14.5 综合评分

| 维度 | 原版 | 优化版 | 评价 |
|------|------|--------|------|
| 启动速度 | ★★☆☆☆ | ★★★★★ | 生产可用 |
| 模型体积 | ★★☆☆☆ | ★★★★☆ | SD卡友好 |
| 内存占用 | ★★★☆☆ | ★★★★☆ | 有余量跑其他服务 |
| 推理速度 | ★★★★☆ | ★★★☆☆ | 可接受（慢0.8s） |
| 音频质量 | ★★★★☆ | ★★★★☆ | 完全一致 |
| 可维护性 | ★★★★★ | ★★★☆☆ | 多了一些依赖文件 |

---

## 十五、推理加速：HiFiGAN ONNX 导出（3倍提升）

> 2026-06-09 更新：将声码器 HiFiGAN 导出为 ONNX FP32 格式，推理速度提升 3 倍。

### 15.1 为什么 HiFiGAN 是瓶颈？

通过对推理各阶段的精确计时，发现：

| 阶段 | 12字句耗时 | 占比 |
|------|-----------|------|
| Frontend (jieba+G2P+BERT) | 0.06s | 1% |
| FastSpeech2 (声学模型) | 0.47s | 11% |
| **HiFiGAN (声码器)** | **3.76s** | **88%** |

HiFiGAN 吃掉了近 90% 的推理时间！这是因为声码器需要把梅尔频谱逐帧展开为音频采样点——每输入 1 帧频谱，输出 300 个采样点。对于 12 字中文（约 150 帧频谱），HiFiGAN 需要生成 45,000 个采样点，涉及数百次卷积运算。

### 15.2 ONNX 导出方案

PaddlePaddle 的推理引擎在 ARM CPU 上并非最优。ONNX Runtime 对 CPU 推理有更好的图优化。将 HiFiGAN 导出为 ONNX：

```python
import paddle

# 导出 generator.forward（输入 [B, 80, T]，输出 [B, 1, T×300]）
paddle.onnx.export(
    generator,
    "hifigan_forward.onnx",
    input_spec=[paddle.static.InputSpec(shape=[1, 80, -1],
                dtype='float32', name='mel')],
    opset_version=14,
)
```

导出后 ONNX Runtime 自动进行常量折叠（constant folding），将 1852 个计算节点优化为 596 个，模型大小 52MB。

### 15.3 INT8 量化的 ARM64 限制

尝试对 HiFiGAN ONNX 进行 INT8 量化：

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("hifigan.onnx", "hifigan_int8.onnx",
                 weight_type=QuantType.QInt8)
```

量化后仅 17.5MB（-66%），但在 ARM64 版 ONNX Runtime 上加载时报错：

```
NotImplemented: Could not find an implementation for ConvInteger(10) node
```

**原因**：INT8 量化依赖 `ConvInteger` 算子，而 ARM64 版 ONNX Runtime 尚未实现该算子。x86 平台（有 VNNI 指令集）可以正常使用 INT8 推理。ARM64 用户暂时只能用 FP32 版本。

### 15.4 加速效果

| 文本 | 帧数 | PaddlePaddle | ONNX FP32 | 加速比 |
|------|------|-------------|-----------|--------|
| "你好" (2字) | 50 | 0.97s | 0.27s | **3.6×** |
| "你好树莓派" (5字) | 71 | 1.34s | 0.45s | **3.0×** |
| 12字句 | 150 | 2.76s | 0.76s | **3.6×** |
| 20字句 | 200 | 3.60s | 1.02s | **3.5×** |

端到端效果（含 Frontend + AM）：

| 文本 | 优化前 | ONNX 后 | 提升 |
|------|--------|---------|------|
| "你好树莓派" | 2.1s | **1.0s** | 2.1× |
| 12字巡检句 | 4.3s | **1.9s** | 2.3× |

### 15.5 集成方式

优化版 Web 服务自动检测 ONNX 模型：

```python
class FastTTS:
    def load_models(self):
        # ... 加载 Paddle 模型 ...
        
        # 尝试加载 ONNX 加速（可选）
        onnx_path = QUANT_DIR / "hifigan_forward.onnx"
        if onnx_path.exists():
            self.voc_sess_onnx = ort.InferenceSession(onnx_path)
    
    def synthesize(self, text, output):
        mel = self.am_inference(phones)      # Paddle AM
        if self.voc_sess_onnx:
            # ONNX 快速路径（3x faster）
            c = norm_mel.T.unsqueeze(0)
            wav = self.voc_sess_onnx.run(None, {'mel': c.numpy()})
        else:
            # Paddle 回退
            wav = self.voc_inference(mel)
```

### 15.6 累积优化成果

| 指标 | Day 1 (原始) | Day 2 (最终) | 总提升 |
|------|-------------|-------------|--------|
| 模型加载 | 55s | 15s | **3.7×** |
| 推理 (12字) | 4.3s | 1.9s | **2.3×** |
| 首次可用 | 60s | 17s | **3.5×** |
| 模型体积 | 2.6GB | 396MB | **-85%** |
| 内存占用 | 2.8GB | 1.5GB | **-46%** |

---

> 全文完。如果你也在树莓派上部署PaddleSpeech遇到了其他问题，欢迎留言交流。

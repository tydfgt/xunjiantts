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

> 全文完。如果你也在树莓派上部署PaddleSpeech遇到了其他问题，欢迎留言交流。

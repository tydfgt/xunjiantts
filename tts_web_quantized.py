#!/usr/bin/env python3
"""
PaddleSpeech 中文 TTS Web 前端 — 量化优化版
=============================================
相比原版 tts_web.py 的改进:
  1. 模型加载: 55s → ~15s (state_dict 替代 .pdz，加速 3-4x)
  2. G2P 模型: 605MB → 152MB (ONNX INT8 量化，压缩 75%)
  3. 模型权重: 1.1GB → 191MB (去除训练优化器状态)
  4. 首次推理预热，后续请求无冷启动

启动方式:
  conda activate paddlespeech
  python tts_web_quantized.py

浏览器打开: http://<树莓派IP>:8766

原版 tts_web.py 保留不变，端口 8765。
"""

import io
import os
import sys
import time
import uuid
import pickle
import subprocess
from pathlib import Path

import numpy as np
import paddle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

# ===================== 路径配置 =====================
PROJECT_DIR = Path.home() / "tts"
QUANT_DIR = PROJECT_DIR / "quantized_models"
OUTPUT_DIR = Path("/tmp/tts_web_quantized_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# USB 声卡播放设备
PLAYBACK_DEVICES = ["plughw:2,0", "plughw:3,0", "plughw:4,0", "default"]

app = FastAPI(title="TTS 语音合成 (量化优化版)")

# 全局变量
tts_engine = None


# ===================== 快速模型加载 =====================

class FastTTS:
    """
    快速 TTS 引擎：跳过 .pdz 解析，使用预提取的 state_dict 加载模型。
    加载时间从 ~55s 降至 ~15s。
    """

    def __init__(self):
        self.am_inference = None
        self.voc_inference = None
        self.frontend = None
        self.am_config = None
        self.voc_config = None
        self.voc_norm = None          # ZScore normalizer (用于 ONNX 推理)
        self.voc_sess_onnx = None     # ONNX Runtime session (HiFiGAN FP32)

    def load_models(self):
        """加载所有模型（AM + Vocoder + Frontend）"""
        from paddlespeech.t2s.models.fastspeech2 import FastSpeech2, FastSpeech2Inference
        from paddlespeech.t2s.models.hifigan import HiFiGANGenerator, HiFiGANInference
        from paddlespeech.t2s.modules.normalizer import ZScore
        from omegaconf import OmegaConf

        print("[快速加载] 开始加载模型...")
        t_total = time.time()

        # ---- 加载 normalizer 参数 ----
        with open(QUANT_DIR / "normalizer_params.pkl", 'rb') as f:
            norm_params = pickle.load(f)

        # ---- FastSpeech2 (声学模型) ----
        t0 = time.time()
        am_model_dir = Path(os.path.expanduser(
            "~/.paddlespeech/models/fastspeech2_csmsc-zh/1.0/"
            "fastspeech2_nosil_baker_ckpt_0.4"))
        am_config = OmegaConf.load(str(list(am_model_dir.rglob("default.yaml"))[0]))

        phones_dict_path = am_model_dir / "phone_id_map.txt"
        with open(phones_dict_path) as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)

        fs2 = FastSpeech2(idim=vocab_size, odim=am_config.n_mels,
                          **am_config["model"])
        am_norm = ZScore(
            mu=paddle.to_tensor(norm_params['am']['mu']),
            sigma=paddle.to_tensor(norm_params['am']['sigma']))
        self.am_inference = FastSpeech2Inference(normalizer=am_norm, model=fs2)
        self.am_inference.eval()

        am_sd_path = QUANT_DIR / "fastspeech2_full.pdparams"
        if am_sd_path.exists():
            self.am_inference.set_state_dict(paddle.load(str(am_sd_path)))
        else:
            raise FileNotFoundError(f"AM state_dict 不存在: {am_sd_path}")
        print(f"  [AM] FastSpeech2 加载完成 ({time.time()-t0:.1f}s)")

        # ---- HiFiGAN (声码器) ----
        t0 = time.time()
        voc_model_dir = Path(os.path.expanduser(
            "~/.paddlespeech/models/hifigan_csmsc-zh/1.0/"
            "hifigan_csmsc_ckpt_0.1.1"))
        voc_config = OmegaConf.load(str(list(voc_model_dir.rglob("default.yaml"))[0]))

        hfg = HiFiGANGenerator(**voc_config["generator_params"])
        voc_norm = ZScore(
            mu=paddle.to_tensor(norm_params['voc']['mu']),
            sigma=paddle.to_tensor(norm_params['voc']['sigma']))
        self.voc_inference = HiFiGANInference(
            normalizer=voc_norm, hifigan_generator=hfg)
        self.voc_inference.eval()

        voc_sd_path = QUANT_DIR / "hifigan_full.pdparams"
        if voc_sd_path.exists():
            # 关键：先移除 weight_norm 再加载权重！
            # 官方代码也是 set_state_dict 后调用 remove_weight_norm，
            # 但官方加载的 .pdz 已包含正确的 weight_g/weight_v。
            # 我们的 state_dict 只有 raw weight（从已 remove_weight_norm
            # 的模型保存），所以必须先移除 weight_norm，再加载。
            self._remove_all_weight_norm(hfg)
            self.voc_inference.set_state_dict(paddle.load(str(voc_sd_path)))
        else:
            raise FileNotFoundError(f"VOC state_dict 不存在: {voc_sd_path}")
        t_voc = time.time() - t0
        print(f"  [VOC] HiFiGAN 加载完成 ({t_voc:.1f}s)")

        # ---- ONNX HiFiGAN (可选加速) ----
        self.voc_norm = voc_norm
        onnx_path = QUANT_DIR / "hifigan_forward.onnx"
        if onnx_path.exists():
            t0 = time.time()
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            self.voc_sess_onnx = ort.InferenceSession(
                str(onnx_path), providers=['CPUExecutionProvider'],
                sess_options=opts)
            t_onnx = time.time() - t0
            print(f"  [ONNX] HiFiGAN ONNX 加载完成 ({t_onnx:.1f}s, ~3x 加速推理)")
        else:
            print(f"  [ONNX] 未找到 ONNX 模型，使用 Paddle 推理")

        # ---- Frontend (文本前端) ----
        t0 = time.time()
        from paddlespeech.t2s.frontend.zh_frontend import Frontend
        self.frontend = Frontend(
            phone_vocab_path=str(phones_dict_path),
            tone_vocab_path=None)
        t_fe = time.time() - t0
        print(f"  [Frontend] 文本前端加载完成 ({t_fe:.1f}s)")

        # 保存配置
        self.am_config = am_config
        self.voc_config = voc_config

        t_total_elapsed = time.time() - t_total
        print(f"[快速加载] ✅ 全部就绪，总耗时 {t_total_elapsed:.1f}s")
        print(f"   (原版 .pdz 加载约需 55s, 加速 {55/t_total_elapsed:.1f}x)")

    def _remove_all_weight_norm(self, module):
        """移除所有 weight_norm。官方 get_voc_inference 在加载权重后会调用
        remove_weight_norm()，推理时不需要 weight_norm。"""
        import paddle.nn.utils as nn_utils
        for name, child in module.named_children():
            self._remove_all_weight_norm(child)
        if hasattr(module, 'weight_g'):
            try:
                nn_utils.remove_weight_norm(module)
            except Exception:
                pass

    def synthesize(self, text: str, output_path: str):
        """合成语音到文件"""
        from paddlespeech.t2s.exps.syn_utils import run_frontend

        # --- Frontend：文本 → 音素序列 ---
        frontend_dict = run_frontend(
            frontend=self.frontend,
            text=text,
            merge_sentences=False,
            get_tone_ids=False,
            lang='zh',
            to_tensor=True,
        )
        phone_ids = frontend_dict['phone_ids']

        # --- AM：音素 → 梅尔频谱 / VOC：梅尔频谱 → 音频 ---
        wav_all = None
        for part_phone_ids in phone_ids:
            with paddle.no_grad():
                mel = self.am_inference(part_phone_ids)  # [T, 80]

            if self.voc_sess_onnx is not None:
                # ONNX 加速路径 (~3x faster than Paddle)
                norm_mel = self.voc_norm(mel)  # 归一化
                c = norm_mel.transpose([1, 0]).unsqueeze(0)  # [1, 80, T]
                onnx_out = self.voc_sess_onnx.run(
                    None, {'mel': c.numpy().astype('float32')})
                wav = paddle.to_tensor(
                    onnx_out[0].squeeze(0).transpose(1, 0))  # [N, 1]
            else:
                # Paddle 推理（回退方案）
                wav = self.voc_inference(mel)

            if wav_all is None:
                wav_all = wav
            else:
                wav_all = paddle.concat([wav_all, wav])

        # --- 保存音频 ---
        import soundfile as sf
        wav_np = wav_all.numpy().flatten()
        sf.write(output_path, wav_np, samplerate=self.am_config.fs)

        return wav_np


# ===================== 音频播放 =====================

def play_on_usb_speaker(filepath: str) -> str:
    """通过 USB 喇叭播放音频"""
    for dev in PLAYBACK_DEVICES:
        try:
            result = subprocess.run(
                ["aplay", "-D", dev, filepath],
                capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                return f"USB喇叭 ({dev})"
        except Exception:
            continue
    return "播放失败"


# ===================== 前端页面 =====================

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🎙️ TTS 语音合成 (量化版)</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    min-height: 100vh; color: #e0e0e0;
    display: flex; justify-content: center; align-items: center;
  }
  .container {
    width: 100%; max-width: 600px; padding: 30px;
    background: rgba(255,255,255,0.06);
    border-radius: 20px; backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
  }
  h1 { text-align: center; font-size: 1.8em; margin-bottom: 4px; }
  .subtitle { text-align: center; color: #888; margin-bottom: 24px; font-size: 0.85em; }
  .badge {
    display: inline-block; background: #43a047; color: #fff;
    padding: 2px 8px; border-radius: 10px; font-size: 0.75em;
    margin-left: 4px; vertical-align: middle;
  }
  textarea {
    width: 100%; height: 120px; padding: 16px;
    font-size: 1.1em; border-radius: 12px; border: 1px solid rgba(255,255,255,0.2);
    background: rgba(255,255,255,0.08); color: #fff;
    resize: vertical; outline: none; transition: border-color .3s;
  }
  textarea:focus { border-color: #4fc3f7; }
  .controls { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
  button {
    flex: 1; min-width: 100px; padding: 14px 24px;
    font-size: 1em; font-weight: 600; border: none; border-radius: 12px;
    cursor: pointer; transition: all .2s; color: #fff;
  }
  .btn-synth { background: linear-gradient(135deg, #43a047, #66bb6a); }
  .btn-synth:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(67,160,71,.4); }
  .btn-synth:disabled { opacity: .5; cursor: not-allowed; transform: none; box-shadow: none; }
  .btn-quick { background: rgba(255,255,255,0.1); font-size: 0.85em; padding: 10px 16px; }
  .btn-quick:hover { background: rgba(255,255,255,0.2); }
  .status {
    margin-top: 20px; padding: 12px 16px; border-radius: 10px;
    text-align: center; font-size: 0.95em; display: none;
  }
  .status.info { display: block; background: rgba(79,195,247,0.15); color: #4fc3f7; }
  .status.ok { display: block; background: rgba(102,187,106,0.15); color: #81c784; }
  .status.err { display: block; background: rgba(229,57,53,0.15); color: #ef5350; }
  audio {
    width: 100%; margin-top: 16px; outline: none;
    border-radius: 10px; display: none;
  }
  audio.show { display: block; }
  .history { margin-top: 24px; }
  .history h3 { color: #888; font-size: 0.9em; margin-bottom: 8px; }
  .history-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; margin-bottom: 6px;
    background: rgba(255,255,255,0.05); border-radius: 8px;
    cursor: pointer; transition: background .2s;
  }
  .history-item:hover { background: rgba(255,255,255,0.12); }
  .history-item .text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.9em; }
  .history-item .time { color: #666; font-size: 0.75em; white-space: nowrap; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,.3);
    border-top-color: #fff; border-radius: 50%;
    animation: spin .8s linear infinite; vertical-align: middle; margin-right: 6px;
  }
</style>
</head>
<body>
<div class="container">
  <h1>🎙️ TTS 语音合成 <span class="badge">量化版</span></h1>
  <p class="subtitle">FastSpeech2 + HiFiGAN · INT8 G2P · 快速加载 · 树莓派5</p>

  <textarea id="textInput" placeholder="请输入要合成的中文文本...">你好树莓派，我是你的语音助手。</textarea>

  <div class="controls">
    <button class="btn-synth" id="btnSynth" onclick="synthesize()">🔊 合成并播放</button>
  </div>

  <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
    <button class="btn-quick" onclick="quickText('你好，我是树莓派语音助手。')">👋 问候</button>
    <button class="btn-quick" onclick="quickText('前方发现障碍物，请注意避让。')">⚠️ 告警</button>
    <button class="btn-quick" onclick="quickText('巡检任务已完成，一切正常。')">✅ 完成</button>
    <button class="btn-quick" onclick="quickText('温度三十七点五度，湿度百分之六十。')">🌡️ 温湿度</button>
  </div>

  <div id="status" class="status"></div>
  <audio id="audioPlayer" controls></audio>

  <div class="history" id="history">
    <h3>📝 合成历史</h3>
    <div id="historyList"></div>
  </div>
</div>

<script>
let synthHistory = [];

function setStatus(msg, type) {
  const el = document.getElementById('status');
  el.className = 'status ' + type;
  el.innerHTML = msg;
}

function quickText(text) {
  document.getElementById('textInput').value = text;
  synthesize();
}

async function synthesize() {
  const text = document.getElementById('textInput').value.trim();
  if (!text) { setStatus('请输入文本', 'err'); return; }
  if (text.length > 200) { setStatus('文本过长，最多200字', 'err'); return; }

  const btn = document.getElementById('btnSynth');
  const audio = document.getElementById('audioPlayer');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>合成中...';
  audio.classList.remove('show');
  setStatus('正在合成语音，请稍候...', 'info');

  const t0 = Date.now();
  try {
    const resp = await fetch('/api/tts', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text})
    });
    const data = await resp.json();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    if (!resp.ok) {
      setStatus(`错误: ${data.detail || '未知错误'}`, 'err');
      btn.disabled = false;
      btn.innerHTML = '🔊 合成并播放';
      return;
    }

    const audioUrl = `/api/audio/${data.file_id}`;
    audio.src = audioUrl;
    audio.classList.add('show');
    audio.load();
    audio.play().catch(e => console.log('自动播放被阻止，请手动点击播放'));

    setStatus(`✅ 合成 ${data.elapsed}秒 · ${data.size_kb}KB · ${data.speaker}`, 'ok');
    addHistory(text, data.file_id, elapsed);
  } catch (e) {
    setStatus(`网络错误: ${e.message}`, 'err');
  }
  btn.disabled = false;
  btn.innerHTML = '🔊 合成并播放';
}

function addHistory(text, fileId, elapsed) {
  synthHistory.unshift({text, fileId, elapsed, time: new Date().toLocaleTimeString()});
  if (synthHistory.length > 10) synthHistory.pop();
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('historyList');
  list.innerHTML = synthHistory.map((h, i) => `
    <div class="history-item" onclick="playHistory('${h.fileId}')">
      <span class="text" title="${h.text}">${h.text}</span>
      <span class="time">${h.elapsed}s · ${h.time}</span>
    </div>
  `).join('');
}

function playHistory(fileId) {
  const audio = document.getElementById('audioPlayer');
  audio.src = `/api/audio/${fileId}`;
  audio.classList.add('show');
  audio.load();
  audio.play();
}
</script>
</body>
</html>"""


# ===================== API =====================

@app.on_event("startup")
async def startup():
    global tts_engine
    print("=" * 50)
    print("[启动] 初始化量化优化版 TTS 引擎...")
    tts_engine = FastTTS()
    tts_engine.load_models()

    # 预热：跑一次推理让 ONNX Runtime 和 Paddle 完成 JIT 编译
    print("[预热] 执行预热推理...")
    warmup_path = OUTPUT_DIR / "_warmup.wav"
    tts_engine.synthesize("预热测试", str(warmup_path))
    if warmup_path.exists():
        warmup_path.unlink()
    print("[启动] ✅ TTS 引擎就绪！")
    print("=" * 50)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/tts")
async def api_tts(request: Request):
    global tts_engine
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"detail": "文本不能为空"}, status_code=400)
    if len(text) > 300:
        return JSONResponse({"detail": "文本最长300字"}, status_code=400)

    file_id = uuid.uuid4().hex[:8]
    output_path = OUTPUT_DIR / f"{file_id}.wav"

    t0 = time.time()
    try:
        tts_engine.synthesize(text, str(output_path))
    except Exception as e:
        return JSONResponse({"detail": f"合成失败: {str(e)}"}, status_code=500)

    synth_time = time.time() - t0
    size_kb = round(os.path.getsize(output_path) / 1024, 1)
    speaker = play_on_usb_speaker(str(output_path))

    print(f"[TTS] 合成 {synth_time:.1f}s, {size_kb}KB, "
          f"喇叭={speaker}, text={text[:40]}...")

    return {
        "file_id": file_id,
        "elapsed": round(synth_time, 1),
        "size_kb": size_kb,
        "speaker": speaker,
    }


@app.get("/api/audio/{file_id}")
async def get_audio(file_id: str):
    path = OUTPUT_DIR / f"{file_id}.wav"
    if not path.exists():
        return JSONResponse({"detail": "文件不存在"}, status_code=404)
    return FileResponse(str(path), media_type="audio/wav")


# ===================== 主入口 =====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TTS Web 量化优化版")
    parser.add_argument("--port", type=int, default=8766,
                        help="服务端口 (默认 8766，原版用 8765)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="绑定地址")
    args = parser.parse_args()

    print(f"启动 TTS Web 量化优化版: http://0.0.0.0:{args.port}")
    print(f"原版 tts_web.py 端口 8765 不受影响")
    uvicorn.run(app, host=args.host, port=args.port,
                log_level="info")

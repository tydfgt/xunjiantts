#!/usr/bin/env python3
"""
PaddleSpeech 中文 TTS Web 前端
================================
启动方式:
  conda activate paddlespeech
  python tts_web.py

然后浏览器打开: http://<树莓派IP>:8765

模型: FastSpeech2 + HiFiGAN, CSMSC baker 中文女声
输出: USB 喇叭自动播放
"""

import io
import os
import time
import uuid
import subprocess
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

from paddlespeech.cli.tts import TTSExecutor

app = FastAPI(title="TTS 语音合成测试")

# 全局 TTS 引擎
tts_engine: TTSExecutor = None
OUTPUT_DIR = Path("/tmp/tts_web_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# USB 声卡播放设备
PLAYBACK_DEVICES = ["plughw:2,0", "plughw:3,0", "plughw:4,0", "default"]


def play_on_usb_speaker(filepath: str) -> str:
    """通过 USB 喇叭播放音频，返回播放结果描述"""
    for dev in PLAYBACK_DEVICES:
        try:
            result = subprocess.run(
                ["aplay", "-D", dev, filepath],
                capture_output=True, text=True, timeout=15
            )
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
<title>🎙️ TTS 语音合成测试</title>
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
  h1 { text-align: center; font-size: 1.8em; margin-bottom: 8px; }
  .subtitle { text-align: center; color: #888; margin-bottom: 24px; font-size: 0.9em; }
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
  .btn-stop { background: linear-gradient(135deg, #e53935, #ef5350); }
  .btn-stop:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(229,57,53,.4); }
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

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
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
  <h1>🎙️ TTS 语音合成</h1>
  <p class="subtitle">PaddleSpeech · FastSpeech2 + HiFiGAN · 中文女声 · 树莓派 5</p>

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

    // 播放音频
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
    print("[启动] 加载中文 TTS 引擎...")
    tts_engine = TTSExecutor()
    print("[启动] TTS 引擎就绪！")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/tts")
async def api_tts(request: Request):
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
        tts_engine(text=text, output=str(output_path))
    except Exception as e:
        return JSONResponse({"detail": f"合成失败: {str(e)}"}, status_code=500)

    synth_time = time.time() - t0
    size_kb = round(os.path.getsize(output_path) / 1024, 1)
    speaker = play_on_usb_speaker(str(output_path))

    print(f"[TTS] 合成{synth_time:.1f}s, {size_kb}KB, "
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
    return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
    print("=" * 50)
    print("  PaddleSpeech TTS Web 前端")
    print("  访问地址: http://192.168.3.108:8765")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")

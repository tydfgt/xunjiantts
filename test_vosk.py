#!/usr/bin/env python3
"""测试 Vosk 离线中文语音识别

支持两种 TTS 方式生成测试音频:
  1. espeak-ng (默认，速度快但声音机械)
  2. PaddleSpeech (声音自然但需额外安装)

用法:
  python test_vosk.py                    # 默认 espeak
  python test_vosk.py --tts paddlespeech # 使用 PaddleSpeech
  python test_vosk.py --tts espeak       # 使用 espeak
"""

import os
import sys
import wave
import json
import argparse
from vosk import Model, KaldiRecognizer

MODEL_PATH = os.path.expanduser("~/tts/models/vosk-model-small-cn-0.22")
TEST_TEXT = "你好树莓派，今天天气不错"
WAV_FILE = "/tmp/test_tts.wav"
SAMPLE_RATE = 16000


def generate_audio_espeak(text: str, output_wav: str):
    """用 espeak-ng 生成测试音频"""
    raw_wav = "/tmp/test_tts_raw.wav"
    print(f"[TTS:espeak] 合成语音: {text}")
    ret = os.system(
        f'espeak-ng -v zh -s 130 -w {raw_wav} "{text}" 2>/dev/null'
    )
    if ret != 0 or not os.path.exists(raw_wav):
        print("[TTS:espeak] 音频生成失败! 请先安装: sudo apt install espeak-ng")
        return False

    ret = os.system(
        f'ffmpeg -y -i {raw_wav} -ar {SAMPLE_RATE} -ac 1 -sample_fmt s16 {output_wav} '
        f'-loglevel quiet 2>/dev/null'
    )
    os.remove(raw_wav)

    if ret == 0 and os.path.exists(output_wav):
        size = os.path.getsize(output_wav)
        print(f"[TTS:espeak] 音频已生成: {output_wav} ({size} bytes, {SAMPLE_RATE}Hz)")
        return True
    else:
        print("[TTS:espeak] 音频转换失败! 请先安装 ffmpeg: sudo apt install ffmpeg")
        return False


def generate_audio_paddlespeech(text: str, output_wav: str):
    """用 PaddleSpeech 生成测试音频（声音自然）"""
    import subprocess

    # 先尝试用 conda run 在 paddlespeech 环境中执行
    raw_output = "/tmp/test_tts_paddle.wav"

    print(f"[TTS:paddle] 合成语音: {text}")
    result = subprocess.run(
        [
            "conda", "run", "-n", "paddlespeech", "--no-capture-output",
            "paddlespeech", "tts",
            "--input", text,
            "--output", raw_output,
        ],
        capture_output=True, text=True, timeout=120
    )

    if result.returncode != 0 or not os.path.exists(raw_output):
        # 尝试直接调用（当前环境已安装 PaddleSpeech）
        print("[TTS:paddle] conda run 失败，尝试直接调用...")
        result2 = subprocess.run(
            ["paddlespeech", "tts", "--input", text, "--output", raw_output],
            capture_output=True, text=True, timeout=120
        )
        if result2.returncode != 0 or not os.path.exists(raw_output):
            print(f"[TTS:paddle] PaddleSpeech 调用失败!")
            print(f"  stderr: {result.stderr[:200] if result.stderr else 'N/A'}")
            return False

    # 转换采样率到 16000Hz (Vosk 要求)
    ret = os.system(
        f'ffmpeg -y -i {raw_output} -ar {SAMPLE_RATE} -ac 1 -sample_fmt s16 {output_wav} '
        f'-loglevel quiet 2>/dev/null'
    )
    os.remove(raw_output)

    if ret == 0 and os.path.exists(output_wav):
        size = os.path.getsize(output_wav)
        print(f"[TTS:paddle] 音频已生成: {output_wav} ({size} bytes, {SAMPLE_RATE}Hz)")
        return True
    else:
        print("[TTS:paddle] 音频转换失败!")
        return False


def generate_test_audio(tts_engine: str = "espeak"):
    """生成测试音频"""
    if tts_engine == "paddlespeech":
        return generate_audio_paddlespeech(TEST_TEXT, WAV_FILE)
    else:
        return generate_audio_espeak(TEST_TEXT, WAV_FILE)

def recognize_speech():
    """用 Vosk 识别音频"""
    if not os.path.exists(MODEL_PATH):
        print(f"[ASR] 模型不存在: {MODEL_PATH}")
        return

    print(f"[ASR] 加载模型中...")
    model = Model(MODEL_PATH)
    
    wf = wave.open(WAV_FILE, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
        print(f"[ASR] 音频格式需为单声道16-bit {SAMPLE_RATE}Hz，当前格式: "
              f"channels={wf.getnchannels()}, sampwidth={wf.getsampwidth()}, "
              f"framerate={wf.getframerate()}")
        wf.close()
        return

    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    result = json.loads(rec.FinalResult())
    recognized = result.get("text", "")
    
    print(f"[ASR] 原始文本: {TEST_TEXT}")
    print(f"[ASR] 识别结果: {recognized}")

    wf.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vosk 离线中文语音识别测试")
    parser.add_argument("--tts", choices=["espeak", "paddlespeech"], default="espeak",
                        help="TTS 引擎: espeak (默认) 或 paddlespeech")
    args = parser.parse_args()

    if generate_test_audio(args.tts):
        recognize_speech()

#!/usr/bin/env python3
"""
PaddleSpeech 中文 TTS 测试脚本 — 树莓派 5
=============================================
功能:
  1. 离线中文语音合成（女声，接近真人）
  2. 支持直接播放或保存为 WAV 文件
  3. 支持多种语速和音量调节

依赖:
  conda activate paddlespeech

首次运行会自动从网络下载模型（~300MB），后续使用无需联网。

模型说明:
  - 默认: FastSpeech2 + Parallel WaveGAN (中文女声, baker 数据集训练)
  - 声音自然度远好于 espeak
  - 树莓派 5 上合成一句话约 1-3 秒
"""

import os
import sys
import time
import argparse

# ---------- 配置 ----------
OUTPUT_DIR = os.path.expanduser("~/tts/output")
DEFAULT_TEXT = "你好树莓派，我是你的语音助手，今天天气不错，适合出去走走。"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def tts_paddlespeech_cli(text: str, output: str, speed: float = 1.0):
    """
    使用 PaddleSpeech CLI 进行语音合成。
    这是最简单可靠的方式，适合大部分场景。
    """
    import subprocess

    cmd = [
        "paddlespeech", "tts",
        "--input", text,
        "--output", output,
    ]
    # 注意: PaddleSpeech CLI 的语速参数因版本而异
    # 部分版本支持 --speed 参数

    print(f"[TTS] 合成中: {text[:50]}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[TTS] 错误:\n{result.stderr}")
        return False

    print(f"[TTS] 合成完成，耗时 {elapsed:.1f}s")
    if os.path.exists(output):
        size_kb = os.path.getsize(output) / 1024
        print(f"[TTS] 输出文件: {output} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"[TTS] 输出文件未生成: {output}")
        return False


def tts_paddlespeech_api(text: str, output: str):
    """
    使用 PaddleSpeech Python API 进行语音合成。
    更灵活，可控制更多参数。
    """
    try:
        from paddlespeech.cli.tts import TTSExecutor
    except ImportError:
        print("[TTS] PaddleSpeech 未安装, 请先运行 install_paddlespeech.sh")
        return False

    tts = TTSExecutor()

    print(f"[TTS] 合成中: {text[:50]}...")
    t0 = time.time()
    tts(text=text, output=output)
    elapsed = time.time() - t0

    print(f"[TTS] 合成完成，耗时 {elapsed:.1f}s")
    if os.path.exists(output):
        size_kb = os.path.getsize(output) / 1024
        print(f"[TTS] 输出文件: {output} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"[TTS] 输出文件未生成: {output}")
        return False


def play_audio(filepath: str):
    """使用 aplay 播放音频（树莓派 USB 声卡）"""
    import subprocess
    print(f"[播放] {filepath}")
    # 依次尝试 USB 声卡 (plughw:2,0 和 plughw:3,0)
    for dev in ["plughw:2,0", "plughw:3,0", "default"]:
        ret = subprocess.run(["aplay", "-D", dev, filepath],
                            capture_output=True, timeout=10)
        if ret.returncode == 0:
            print(f"[播放] 设备 {dev} 播放成功")
            return True
    print("[播放] 所有设备播放失败，请检查 aplay -l")
    return False


def batch_synthesis(texts: list, method: str = "cli"):
    """
    批量合成多段文本。
    可用于预先合成常用语音，后续直接播放。
    """
    ensure_output_dir()
    results = []

    for i, text in enumerate(texts):
        output = os.path.join(OUTPUT_DIR, f"batch_{i:03d}.wav")
        if method == "api":
            ok = tts_paddlespeech_api(text, output)
        else:
            ok = tts_paddlespeech_cli(text, output)
        results.append((text, output, ok))

    # 打印汇总
    success = sum(1 for _, _, ok in results if ok)
    print(f"\n[汇总] {success}/{len(results)} 条合成成功")
    for text, output, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {text[:40]} -> {output}")

    return results


# ---------- 测试用例 ----------
TEST_SENTENCES = [
    "你好，我是树莓派语音助手。",
    "前方发现障碍物，请注意避让。",
    "巡检任务已完成，一切正常。",
    "温度三十七点三度，湿度百分之六十五。",
    "正在启动语音广播系统。",
]


def main():
    parser = argparse.ArgumentParser(description="PaddleSpeech 中文 TTS 测试")
    parser.add_argument("--text", "-t", type=str, default=DEFAULT_TEXT,
                        help="要合成的文本")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出 WAV 文件路径")
    parser.add_argument("--play", "-p", action="store_true",
                        help="合成后直接播放")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="批量合成测试用例")
    parser.add_argument("--method", "-m", choices=["cli", "api"], default="cli",
                        help="合成方式: cli (命令行) 或 api (Python接口)")
    args = parser.parse_args()

    ensure_output_dir()

    # --- 单句合成 ---
    if not args.batch:
        if args.output:
            output = args.output
        else:
            output = os.path.join(OUTPUT_DIR, "test_output.wav")

        if args.method == "api":
            ok = tts_paddlespeech_api(args.text, output)
        else:
            ok = tts_paddlespeech_cli(args.text, output)

        if ok and args.play:
            play_audio(output)
        return

    # --- 批量合成 ---
    print(f"[批量] 共 {len(TEST_SENTENCES)} 条测试语句")
    batch_synthesis(TEST_SENTENCES, method=args.method)


if __name__ == "__main__":
    main()

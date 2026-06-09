#!/usr/bin/env python3
"""
PaddleSpeech TTS 量化实验脚本
==============================
目标：对 FastSpeech2 + HiFiGAN ONNX 模型进行 INT8 量化
     减小模型体积、加速推理、降低内存占用

策略：
  1. 通过 PaddleSpeech 内置 ONNX 路径加载模型
  2. 使用 ONNX Runtime dynamic quantization 量化到 INT8
  3. 对比原始 vs 量化后的性能

用法：
  conda activate paddlespeech
  python quantize_experiment.py

原版文件不受影响，所有量化产物放在 ~/tts/quantized_models/ 下。
"""

import os
import sys
import time
import shutil
import argparse
import tempfile
import json
from pathlib import Path
import numpy as np

# 项目路径
PROJECT_DIR = Path.home() / "tts"
QUANTIZED_DIR = PROJECT_DIR / "quantized_models"
QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)

# 测试文本
TEST_SENTENCES = [
    "你好树莓派",
    "前方发现障碍物，请注意避让。",
    "巡检任务已完成，一切正常。",
    "温度三十七点三度，湿度百分之六十五。",
]


# ============================================================
# 第一步：获取 ONNX 模型
# ============================================================

def get_onnx_models():
    """
    通过 PaddleSpeech TTSExecutor 触发 ONNX 模型下载，
    返回 (am_onnx_path, voc_onnx_path)
    """
    from paddlespeech.cli.tts import TTSExecutor

    print("=" * 60)
    print("步骤 1: 获取 ONNX 模型")
    print("=" * 60)

    tts = TTSExecutor()

    # 通过 ONNX 路径初始化，触发模型下载
    print("正在下载/加载 ONNX 模型（首次需要下载，约 200-400MB）...")
    t0 = time.time()
    tts._init_from_path_onnx(
        am='fastspeech2_csmsc',
        voc='hifigan_csmsc',
        lang='zh',
        device='cpu',
        cpu_threads=2,
    )
    elapsed = time.time() - t0
    print(f"ONNX 模型加载完成，耗时 {elapsed:.1f}s")

    # 找到实际的 ONNX 模型文件路径
    am_dir = tts.am_res_path
    voc_dir = tts.voc_res_path

    print(f"\nAM (FastSpeech2) 目录: {am_dir}")
    print(f"VOC (HiFiGAN) 目录: {voc_dir}")

    # 查找 .onnx 文件
    am_onnx = None
    voc_onnx = None
    for f in Path(am_dir).rglob("*.onnx"):
        am_onnx = str(f)
        print(f"  发现 AM ONNX: {f.name} ({os.path.getsize(f)/1024/1024:.1f}MB)")
    for f in Path(voc_dir).rglob("*.onnx"):
        voc_onnx = str(f)
        print(f"  发现 VOC ONNX: {f.name} ({os.path.getsize(f)/1024/1024:.1f}MB)")

    # 也找找 frontend ONNX
    frontend_onnx = None
    for f in Path(am_dir).rglob("*.onnx"):
        if 'frontend' in f.name.lower() or 'g2p' in f.name.lower():
            frontend_onnx = str(f)

    return tts, am_onnx, voc_onnx, frontend_onnx


# ============================================================
# 第二步：量化 ONNX 模型
# ============================================================

def quantize_onnx_model(model_path: str, output_path: str, model_name: str,
                        quant_type: str = "int8"):
    """
    使用 ONNX Runtime 对 ONNX 模型进行量化。

    quant_type:
      - "int8":   动态 INT8 量化（推荐，不需要校准数据）
      - "int8_static": 静态 INT8 量化（需要校准数据，精度更高）
      - "fp16":   FP16 量化（精度损失小，但 ARM CPU 加速有限）
    """
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
    from onnxruntime.quantization.calibrate import CalibrationDataReader
    import onnx

    print(f"\n{'='*60}")
    print(f"量化模型: {model_name}")
    print(f"  输入: {model_path}")
    print(f"  输出: {output_path}")
    print(f"  方式: {quant_type}")
    print(f"{'='*60}")

    # 加载原始模型查看信息
    model = onnx.load(model_path)
    model_size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"  原始大小: {model_size_mb:.1f}MB")
    print(f"  Opset: {model.opset_import[0].version}")

    t0 = time.time()

    if quant_type == "int8":
        # 动态 INT8 量化：在推理时动态量化权重
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
            optimize_model=True,
            extra_options={
                'ActivationSymmetric': True,
                'WeightSymmetric': True,
            }
        )
    elif quant_type == "int8_static":
        # 静态 INT8 量化：使用校准数据
        print("  静态量化需要校准数据，正在生成...")
        # 简化：使用随机数据作为校准
        class RandomDataReader(CalibrationDataReader):
            def __init__(self, model_path, num_samples=20):
                import onnxruntime as ort
                self.num_samples = num_samples
                self.idx = 0
                sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.inputs = []
                for _ in range(num_samples):
                    sample = {}
                    for inp in sess.get_inputs():
                        shape = [max(1, d) if isinstance(d, int) else 1 for d in inp.shape]
                        if inp.type == 'tensor(int64)':
                            sample[inp.name] = np.random.randint(0, 300, shape).astype(np.int64)
                        else:
                            sample[inp.name] = np.random.randn(*shape).astype(np.float32)
                    self.inputs.append(sample)

            def get_next(self):
                if self.idx >= self.num_samples:
                    return None
                ret = self.inputs[self.idx]
                self.idx += 1
                return ret

        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=RandomDataReader(model_path),
            quant_format=QuantType.QInt8,
            optimize_model=True,
        )

    elif quant_type == "fp16":
        # FP16 量化（ONNX 格式转换）
        from onnxruntime.quantization import quantize_fp16
        # 注意：ARM CPU 可能不支持 FP16 推理加速
        print("  注意：FP16 在 ARM CPU 上可能没有加速效果")
        from onnxconverter_common import float16
        model_fp16 = float16.convert_float_to_float16(model)
        import onnx
        onnx.save(model_fp16, output_path)

    elapsed = time.time() - t0
    q_size_mb = os.path.getsize(output_path) / 1024 / 1024
    compression = (1 - q_size_mb / model_size_mb) * 100

    print(f"  量化完成，耗时 {elapsed:.1f}s")
    print(f"  量化后大小: {q_size_mb:.1f}MB (压缩 {compression:.1f}%)")

    return True


# ============================================================
# 第三步：性能对比测试
# ============================================================

class QuantizedTTS:
    """使用量化后的 ONNX 模型进行 TTS 推理"""

    def __init__(self, am_quantized_path: str, voc_quantized_path: str,
                 original_tts):
        import onnxruntime as ort
        self.am_sess = ort.InferenceSession(
            am_quantized_path,
            providers=['CPUExecutionProvider'],
            sess_options=self._make_session_options()
        )
        self.voc_sess = ort.InferenceSession(
            voc_quantized_path,
            providers=['CPUExecutionProvider'],
            sess_options=self._make_session_options()
        )
        # 复用原 TTSExecutor 的 frontend（文本前端不需要量化）
        self.original_tts = original_tts
        self.frontend = original_tts.frontend
        self.am_config = original_tts.am_config

    def _make_session_options(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    def synthesize(self, text: str, output: str = None) -> dict:
        """合成语音，返回性能指标"""
        from paddlespeech.t2s.exps.syn_utils import run_frontend
        import soundfile as sf
        import paddle

        am_name = 'fastspeech2'

        # --- Frontend（文本转音素）---
        t0 = time.time()
        frontend_dict = run_frontend(
            frontend=self.frontend,
            text=text,
            merge_sentences=False,
            get_tone_ids=False,
            lang='zh',
            to_tensor=False,
        )
        frontend_time = time.time() - t0
        phone_ids = frontend_dict['phone_ids']

        # --- AM (音素→梅尔频谱) ---
        am_time = 0
        voc_time = 0
        wav_all = None

        for i, part_phone_ids in enumerate(phone_ids):
            am_st = time.time()
            am_input = {'text': np.array(part_phone_ids, dtype=np.int64)}
            mel = self.am_sess.run(None, am_input)[0]
            am_time += time.time() - am_st

            # --- Vocoder (梅尔频谱→音频) ---
            voc_st = time.time()
            wav = self.voc_sess.run(None, {'logmel': mel})[0]
            voc_time += time.time() - voc_st

            if wav_all is None:
                wav_all = wav
            else:
                wav_all = np.concatenate([wav_all, wav])

        total_time = time.time() - t0

        # 保存音频
        if output and wav_all is not None:
            sf.write(output, wav_all.flatten(), samplerate=self.am_config.fs)

        return {
            'frontend_time': frontend_time,
            'am_time': am_time,
            'voc_time': voc_time,
            'total_time': total_time,
            'audio_samples': len(wav_all.flatten()) if wav_all is not None else 0,
            'sample_rate': self.am_config.fs,
        }


def benchmark_original(tts_executor, texts: list) -> list:
    """测试原始 PaddleSpeech (动态图) 性能"""
    results = []
    from paddlespeech.t2s.exps.syn_utils import run_frontend
    import paddle

    for text in texts:
        print(f"  原版测试: {text[:30]}...")
        t0 = time.time()

        # Frontend
        ft0 = time.time()
        frontend_dict = run_frontend(
            frontend=tts_executor.frontend,
            text=text,
            merge_sentences=False,
            get_tone_ids=False,
            lang='zh',
            to_tensor=True,
        )
        frontend_time = time.time() - ft0

        phone_ids = frontend_dict['phone_ids']
        am_time = 0
        voc_time = 0
        wav_all = None

        for part_phone_ids in phone_ids:
            # AM
            am_st = time.time()
            mel = tts_executor.am_inference(part_phone_ids)
            am_time += time.time() - am_st

            # Vocoder
            voc_st = time.time()
            wav = tts_executor.voc_inference(mel)
            voc_time += time.time() - voc_st

            if wav_all is None:
                wav_all = wav
            else:
                wav_all = paddle.concat([wav_all, wav])

        total_time = time.time() - t0
        results.append({
            'frontend_time': frontend_time,
            'am_time': am_time,
            'voc_time': voc_time,
            'total_time': total_time,
        })
        print(f"    总耗时: {total_time:.2f}s (frontend: {frontend_time:.2f}s, "
              f"AM: {am_time:.2f}s, VOC: {voc_time:.2f}s)")

    return results


def benchmark_onnx_original(tts_executor, texts: list) -> list:
    """测试原始 ONNX (未量化) 性能"""
    results = []
    for text in texts:
        print(f"  ONNX原版测试: {text[:30]}...")
        t0 = time.time()

        tts_executor.infer_onnx(text=text, lang='zh')
        # 重新获取时间
        total_time = time.time() - t0
        ft = getattr(tts_executor, 'frontend_time', 0)
        am = getattr(tts_executor, 'am_time', 0)
        voc = getattr(tts_executor, 'voc_time', 0)

        results.append({
            'frontend_time': ft,
            'am_time': am,
            'voc_time': voc,
            'total_time': total_time,
        })
        print(f"    总耗时: {total_time:.2f}s (frontend: {ft:.2f}s, "
              f"AM: {am:.2f}s, VOC: {voc:.2f}s)")

    return results


def benchmark_quantized(qtts: QuantizedTTS, texts: list) -> list:
    """测试量化 ONNX 模型性能"""
    results = []
    for text in texts:
        print(f"  量化版测试: {text[:30]}...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            res = qtts.synthesize(text, output=tmp.name)
        results.append(res)
        print(f"    总耗时: {res['total_time']:.2f}s "
              f"(frontend: {res['frontend_time']:.2f}s, "
              f"AM: {res['am_time']:.2f}s, VOC: {res['voc_time']:.2f}s)")

    return results


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PaddleSpeech TTS 量化实验")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过模型下载（使用已缓存的模型）")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="跳过硬性测试")
    parser.add_argument("--quant-type", default="int8",
                        choices=["int8", "int8_static"],
                        help="量化类型（默认 int8 动态量化）")
    args = parser.parse_args()

    results = {}

    # ---- 步骤 1: 获取 ONNX 模型 ----
    tts, am_onnx, voc_onnx, frontend_onnx = get_onnx_models()

    if not am_onnx or not voc_onnx:
        print("[错误] 找不到 ONNX 模型文件，请检查 PaddleSpeech 安装")
        sys.exit(1)

    # ---- 步骤 2: 量化 ----
    print(f"\n{'='*60}")
    print("步骤 2: ONNX 模型量化")
    print(f"{'='*60}")

    am_quantized = str(QUANTIZED_DIR / f"fastspeech2_csmsc_{args.quant_type}.onnx")
    voc_quantized = str(QUANTIZED_DIR / f"hifigan_csmsc_{args.quant_type}.onnx")

    quantize_onnx_model(am_onnx, am_quantized, "FastSpeech2 (AM)", args.quant_type)
    quantize_onnx_model(voc_onnx, voc_quantized, "HiFiGAN (Vocoder)", args.quant_type)

    # 汇总大小
    print(f"\n{'='*60}")
    print("模型大小对比")
    print(f"{'='*60}")
    am_orig = os.path.getsize(am_onnx) / 1024 / 1024
    am_q = os.path.getsize(am_quantized) / 1024 / 1024
    voc_orig = os.path.getsize(voc_onnx) / 1024 / 1024
    voc_q = os.path.getsize(voc_quantized) / 1024 / 1024
    print(f"  FastSpeech2: {am_orig:.1f}MB → {am_q:.1f}MB (压缩 {(1-am_q/am_orig)*100:.1f}%)")
    print(f"  HiFiGAN:     {voc_orig:.1f}MB → {voc_q:.1f}MB (压缩 {(1-voc_q/voc_orig)*100:.1f}%)")
    print(f"  总计:        {am_orig+voc_orig:.1f}MB → {am_q+voc_q:.1f}MB")

    # ---- 步骤 3: 性能对比 ----
    if not args.skip_benchmark:
        print(f"\n{'='*60}")
        print("步骤 3: 推理性能对比")
        print(f"{'='*60}")

        # 预热：先跑一次让模型加载到内存
        print("预热中...")
        _ = tts(text="预热", output="/tmp/tts_warmup.wav")

        print("\n--- 原版 PaddleSpeech (动态图) ---")
        results['paddle_original'] = benchmark_original(tts, TEST_SENTENCES)

        print("\n--- ONNX 原版 (未量化) ---")
        results['onnx_original'] = benchmark_onnx_original(tts, TEST_SENTENCES)

        print("\n--- ONNX 量化版 (INT8) ---")
        qtts = QuantizedTTS(am_quantized, voc_quantized, tts)
        # 量化版预热
        qtts.synthesize("预热", output="/tmp/tts_q_warmup.wav")
        results['onnx_quantized'] = benchmark_quantized(qtts, TEST_SENTENCES)

        # ---- 汇总 ----
        print(f"\n{'='*60}")
        print("性能汇总（平均每句）")
        print(f"{'='*60}")
        for name, res_list in results.items():
            if not res_list:
                continue
            avg_total = np.mean([r['total_time'] for r in res_list])
            avg_am = np.mean([r['am_time'] for r in res_list])
            avg_voc = np.mean([r['voc_time'] for r in res_list])
            avg_ft = np.mean([r.get('frontend_time', 0) for r in res_list])
            print(f"  {name:20s}: 总计 {avg_total:.2f}s | "
                  f"Frontend {avg_ft:.2f}s | AM {avg_am:.2f}s | VOC {avg_voc:.2f}s")

        # 保存结果
        summary_path = QUANTIZED_DIR / "benchmark_results.json"
        # 转换 numpy 类型
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        with open(summary_path, 'w') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {summary_path}")

    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"  量化模型目录: {QUANTIZED_DIR}")
    print(f"  原版 tts_web.py 未受影响")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

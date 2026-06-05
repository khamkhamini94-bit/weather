"""Export trained model to ONNX and benchmark FP16 inference.

Usage:
    python export.py --weights output/weights/fold_0/best.pth --output model.onnx
    python export.py --weights output/weights/fold_0/best.pth --trt  # build TensorRT engine
"""
import argparse
import time

import numpy as np
import torch

from config import IMAGE_SIZE, NUM_CLASSES
from models.convnext import build_model


def export_onnx(model, output_path, device):
    """Export model to ONNX with dynamic batch."""
    model.eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"ONNX model exported to {output_path}")


def benchmark_pytorch(model, device, n_warmup=20, n_iter=200):
    """Benchmark PyTorch FP32 model."""
    model.eval()
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    for _ in range(n_warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iter * 1000
    print(f"[PyTorch FP32] latency: {elapsed:.2f} ms")


def benchmark_onnx_fp16(onnx_path, n_warmup=20, n_iter=200):
    """Benchmark ONNX Runtime with FP16."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider"],
        provider_options=[{"enable_fp16": True}],
    )
    x = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float16)
    input_name = session.get_inputs()[0].name

    for _ in range(n_warmup):
        _ = session.run(None, {input_name: x})

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = session.run(None, {input_name: x})
    elapsed = (time.perf_counter() - t0) / n_iter * 1000
    print(f"[ONNX FP16]    latency: {elapsed:.2f} ms")


def detect_num_classes(weights_path):
    """Read num_classes from checkpoint weight shape without loading full model."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    neck_weights = [v for k, v in ckpt["model"].items() if k.startswith("neck.") and k.endswith(".weight")]
    return neck_weights[-1].shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="weather_model.onnx", help="ONNX output path")
    parser.add_argument("--trt", action="store_true", help="Also build TensorRT engine (requires trtexec)")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_classes = detect_num_classes(args.weights)
    print(f"Detected {num_classes} classes from checkpoint")

    model = build_model(num_classes=num_classes).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded weights (F1: {ckpt.get('f1', 'N/A')})")

    # Export ONNX
    export_onnx(model, args.output, device)

    # Benchmark
    if args.benchmark and device.type == "cuda":
        benchmark_pytorch(model, device)
        benchmark_onnx_fp16(args.output)

    # TensorRT
    if args.trt:
        import subprocess
        trt_path = args.output.rsplit(".", 1)[0] + ".trt"
        cmd = (
            f"trtexec --onnx={args.output} --fp16 --saveEngine={trt_path} "
            f"--workspace=4096"
        )
        print(f"Building TensorRT engine:\n  {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print(f"TensorRT engine saved to {trt_path}")


if __name__ == "__main__":
    main()

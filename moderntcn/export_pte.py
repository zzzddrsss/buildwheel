"""把 ModernTCN 导出为 ExecuTorch 的 .pte 文件。
流程：
    1. 构造 ModernTCN 模型（随机权重，可选用 --checkpoint 加载）
    2. 结构重参数化（合并卷积分支，推理更快）
    3. torch.export 导出为 ExportedProgram
    4. ExecuTorch 转换（XNNPACK 后端 partitioner）
    5. 可选：PT2E 量化（int8）→ 导出量化版 .pte
    6. 用 ExecuTorch 运行时自检推理
【输入约定】x: [B, C, T] = [1, enc_in, seq_len]
【输出约定】y: [B, pred_len, C]
用法：
    python moderntcn/export_pte.py --output moderntcn.pte
    python moderntcn/export_pte.py --output moderntcn_int8.pte --quantize
    python moderntcn/export_pte.py --checkpoint model.pth --output moderntcn.pte
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
# 允许直接运行本脚本时导入同目录的 model.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from model import ModernTCNConfig, build_model
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 ModernTCN 为 ExecuTorch .pte")
    parser.add_argument("--output", default="moderntcn.pte", help="输出 .pte 路径")
    parser.add_argument("--seq-len", type=int, default=336, help="输入序列长度 T")
    parser.add_argument("--pred-len", type=int, default=96, help="预测长度")
    parser.add_argument("--enc-in", type=int, default=7, help="变量数（通道数 C）")
    parser.add_argument("--checkpoint", default="", help="可选：加载 .pth 权重")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="导出 int8 量化版本（体积更小，速度更快）",
    )
    parser.add_argument(
        "--no-reparam",
        action="store_true",
        help="跳过结构重参数化（默认执行）",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=8,
        help="量化校准批次数量（默认 8）",
    )
    return parser.parse_args(argv)
def build_example_input(config: ModernTCNConfig) -> torch.Tensor:
    """构造示例输入。
    【关键】ModernTCN 的输入是 [B, C, T]（通道在前），
    不是 [B, T, C]。这里必须用正确的形状，否则导出的图是错的。
    """
    return torch.randn(1, config.enc_in, config.seq_len)
def export_float(model: torch.nn.Module, example_input: torch.Tensor, output: Path) -> bool:
    """导出浮点版 .pte。"""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower
    print("[3/5] torch.export 导出 ExportedProgram")
    try:
        exported = torch.export.export(model, (example_input,))
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] torch.export 失败：{type(exc).__name__}: {exc}")
        return False
    print(f"[PASS] 导出成功，图节点数：{len(list(exported.graph.nodes))}")
    print("[4/5] 转换为 ExecuTorch 程序（XNNPACK 后端）")
    try:
        edge_program = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
        )
        et_program = edge_program.to_executorch()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] ExecuTorch 转换失败：{type(exc).__name__}: {exc}")
        return False
    print("[5/5] 保存浮点 .pte")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(et_program.buffer)
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"[PASS] 已保存：{output}（{size_mb:.2f} MB）")
    return True
def export_quantized(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output: Path,
    calib_batches: int = 8,
) -> bool:
    """导出 int8 量化版 .pte（PT2E + XNNPACKQuantizer）。"""
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from executorch.exir import to_edge_transform_and_lower
        from torch.export import export_for_training
    except ImportError as exc:
        print(f"[FAIL] 量化依赖导入失败：{exc}")
        return False
    print("[Q1/4] 构造 PT2E 训练图")
    try:
        pt2e_model = export_for_training(model, (example_input,)).module()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] export_for_training 失败：{type(exc).__name__}: {exc}")
        return False
    print("[Q2/4] 插入量化观察器（XNNPACK 对称量化）")
    try:
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        prepared = quantizer.prepare(pt2e_model)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] quantizer.prepare 失败：{type(exc).__name__}: {exc}")
        return False
    print(f"[Q3/4] 校准（{calib_batches} 批随机数据）")
    try:
        prepared.eval()
        with torch.no_grad():
            for _ in range(calib_batches):
                prepared(torch.randn_like(example_input))
        quantized = quantizer.convert(prepared)
        quantized.eval()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 校准/转换失败：{type(exc).__name__}: {exc}")
        return False
    print("[Q4/4] 导出量化 .pte")
    try:
        q_exported = torch.export.export(quantized, (example_input,))
        q_edge = to_edge_transform_and_lower(
            q_exported,
            partitioner=[XnnpackPartitioner()],
        )
        q_et = q_edge.to_executorch()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 量化图导出失败：{type(exc).__name__}: {exc}")
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(q_et.buffer)
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"[PASS] 量化版已保存：{output}（{size_mb:.2f} MB）")
    return True
def verify_pte(output: Path, config: ModernTCNConfig) -> None:
    """用 ExecuTorch 运行时加载并推理自检。"""
    print("\n[验证] 用 ExecuTorch 运行时加载并推理")
    try:
        from executorch.runtime import Runtime
        runtime = Runtime.get()
        program = runtime.load_program(str(output))
        method = program.load_method("forward")
        test_input = build_example_input(config)
        out = method.execute([test_input])
        print(f"[PASS] 推理成功，输出 shape：{tuple(out[0].shape)}")
        expected = (1, config.pred_len, config.enc_in)
        if tuple(out[0].shape) == expected:
            print(f"[PASS] 输出 shape 符合预期 {expected}")
        else:
            print(f"[WARN] 输出 shape {tuple(out[0].shape)} 与预期 {expected} 不符")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 运行时验证跳过（{type(exc).__name__}: {exc}）")
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=" * 60)
    print("ModernTCN → ExecuTorch .pte 导出")
    print("=" * 60)
    # ---------- 1. 构造模型 ----------
    config = ModernTCNConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=args.enc_in,
    )
    print(
        f"[1/5] 构造模型：seq_len={config.seq_len}, "
        f"pred_len={config.pred_len}, enc_in={config.enc_in}"
    )
    model = build_model(config)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"[FAIL] 权重文件不存在：{ckpt_path}")
            return 1
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[INFO] 已加载权重：缺失 {len(missing)} 项，多余 {len(unexpected)} 项")
    # ---------- 2. 结构重参数化 ----------
    if not args.no_reparam:
        print("[2/5] 结构重参数化（合并双分支卷积）")
        model.structural_reparam()
    else:
        print("[2/5] 跳过结构重参数化")
    model.eval()
    example_input = build_example_input(config)
    print(f"[INFO] 示例输入 shape = {tuple(example_input.shape)}  (B, C, T)")
    output = Path(args.output)
    # ---------- 3~5. 导出浮点版 ----------
    ok = export_float(model, example_input, output)
    if not ok:
        return 1
    # ---------- 可选：量化版 ----------
    if args.quantize:
        print("\n" + "-" * 60)
        print("开始导出 int8 量化版")
        print("-" * 60)
        q_output = output.with_name(output.stem + "_int8.pte")
        q_ok = export_quantized(
            model, example_input, q_output, calib_batches=args.calib_batches
        )
        if q_ok:
            verify_pte(q_output, config)
        else:
            print("[WARN] 量化版导出失败，但浮点版已成功，流程继续")
    # ---------- 验证浮点版 ----------
    verify_pte(output, config)
    print("\n" + "=" * 60)
    print("导出完成 ✅")
    print("=" * 60)
    return 0
if __name__ == "__main__":
    sys.exit(main())
"""校验 out/ 目录下的 .pte 文件（CI 用，替代 YAML heredoc）。
用法：
    python moderntcn/verify_pte.py out
"""
from __future__ import annotations
import sys
from pathlib import Path
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = Path(argv[0]) if argv else Path("out")
    ptes = sorted(out_dir.glob("*.pte"))
    if not ptes:
        print(f"[FAIL] {out_dir} 下没有 .pte 文件")
        return 1
    failed = False
    for p in ptes:
        data = p.read_bytes()
        size_mb = len(data) / 1024 / 1024
        if len(data) < 16:
            print(f"[FAIL] {p.name} 文件过小（{len(data)} 字节），疑似损坏")
            failed = True
            continue
        # .pte 是 flatbuffer，头部含文件标识；末尾通常有 ET 魔数
        head = data[:8].hex()
        has_et_magic = b"ET" in data[-64:]
        print(
            f"[OK] {p.name}: {len(data)} 字节 ({size_mb:.2f} MB), "
            f"头部={head}, 尾部ET魔数={has_et_magic}"
        )
    if failed:
        return 1
    print(f"[PASS] 共 {len(ptes)} 个 .pte 文件校验通过")
    return 0
if __name__ == "__main__":
    sys.exit(main())

"""从 wheelhouse 安装本地 wheel，并自动补齐其声明的运行时依赖。
用法：
    python scripts/install_wheels.py wheelhouse
逻辑：
    1. 扫描 wheelhouse 下的 .whl 文件
    2. 从每个 wheel 的 METADATA 中解析 Requires-Dist
    3. 过滤掉环境标记不满足当前解释器的依赖（如 python_version < "3.11"）
    4. 先用 pip 联网安装这些依赖，再用 --no-index 安装本地 wheel
这样可避免遗漏新增依赖（例如 scikit-learn 1.9 引入的 narwhals）。
"""
from __future__ import annotations
import argparse
import email.parser
import re
import subprocess
import sys
import zipfile
from pathlib import Path
def parse_requires_dist(wheel: Path) -> list[str]:
    """从 wheel 的 METADATA 中提取 Requires-Dist 列表。"""
    with zipfile.ZipFile(wheel) as zf:
        metadata_names = [
            n for n in zf.namelist() if n.endswith(".dist-info/METADATA")
        ]
        if not metadata_names:
            print(f"[WARN] {wheel.name} 中未找到 METADATA，跳过依赖解析")
            return []
        raw = zf.read(metadata_names[0]).decode("utf-8", errors="replace")
    message = email.parser.Parser().parsestr(raw)
    return message.get_all("Requires-Dist") or []
def marker_matches(requirement: str) -> bool:
    """判断依赖的环境标记（; 之后的部分）在当前解释器下是否成立。
    仅处理常见标记；无法解析时保守返回 True（宁可多装不漏装）。
    """
    if ";" not in requirement:
        return True
    marker = requirement.split(";", 1)[1].strip()
    try:
        from packaging.markers import Marker
    except ImportError:
        # 无 packaging 时保守处理：只过滤明显不匹配的 python_version
        match = re.search(r'python_version\s*([<>=!]+)\s*["\']([\d.]+)["\']', marker)
        if not match:
            return True
        op, ver = match.group(1), match.group(2)
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        try:
            cur_t = tuple(int(x) for x in current.split("."))
            tgt_t = tuple(int(x) for x in ver.split("."))
        except ValueError:
            return True
        if op == "<" and not cur_t < tgt_t:
            return False
        if op == "<=" and not cur_t <= tgt_t:
            return False
        if op == ">" and not cur_t > tgt_t:
            return False
        if op == ">=" and not cur_t >= tgt_t:
            return False
        return True
    try:
        return Marker(marker).evaluate()
    except Exception:  # noqa: BLE001 - 标记无法解析时保守放行
        return True
def normalize_requirement(requirement: str) -> str:
    """去掉环境标记，只保留包名与版本约束。"""
    return requirement.split(";", 1)[0].strip()
def collect_dependencies(wheels: list[Path]) -> list[str]:
    """汇总所有 wheel 的运行时依赖（去重、去标记、排除 extras）。"""
    deps: list[str] = []
    seen: set[str] = set()
    for wheel in wheels:
        for requirement in parse_requires_dist(wheel):
            # 跳过 extras 依赖（形如 foo[bar]>=1 中的 extra 部分由 extra == 标记控制）
            if not marker_matches(requirement):
                continue
            normalized = normalize_requirement(requirement)
            if not normalized:
                continue
            key = re.split(r"[\s<>=!~\[;]", normalized, maxsplit=1)[0].lower()
            if key in seen:
                continue
            seen.add(key)
            deps.append(normalized)
    return deps
def build_install_command(house: Path, wheels: list[Path]) -> list[str]:
    """构造本地 wheel 的安装命令。
    关键：必须传完整路径（而非文件名），否则 pip 会按当前工作目录
    查找文件，导致 [Errno 2] No such file or directory。
    """
    return [
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", str(house),
        "--no-deps",
        *[str(w.resolve()) for w in wheels],
    ]

def run(cmd: list[str]) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    return subprocess.call(cmd)
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="安装本地 wheel 并补齐依赖")
    parser.add_argument("wheelhouse", help="wheel 所在目录")
    args = parser.parse_args(argv)
    house = Path(args.wheelhouse)
    wheels = sorted(house.glob("*.whl"))
    if not wheels:
        print(f"[FAIL] {house} 中未找到任何 wheel")
        return 1
    print(f"[INFO] 发现 {len(wheels)} 个 wheel：{[w.name for w in wheels]}")
    deps = collect_dependencies(wheels)
    print(f"[INFO] 解析到 {len(deps)} 个运行时依赖：{deps}")
    if deps:
        rc = run([sys.executable, "-m", "pip", "install", *deps])
        if rc != 0:
            print("[FAIL] 依赖安装失败")
            return rc
    else:
        print("[INFO] 未解析到外部依赖，跳过依赖安装")
    # 安装本地 wheel 本体（--no-deps 避免 pip 再次联网解析）
    # 注意：传完整路径，避免 pip 按当前工作目录找不到文件
    cmd = build_install_command(house, wheels)
    rc = run(cmd)
    if rc != 0:
        print("[FAIL] 本地 wheel 安装失败")
        return rc
    print("[PASS] 本地 wheel 安装完成")
    return 0
if __name__ == "__main__":
    sys.exit(main())
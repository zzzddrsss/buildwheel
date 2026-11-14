"""校验 wheelhouse 目录下的 .whl 产物是否符合预期。
用法：
    python scripts/verify_wheels.py wheelhouse --expect 2
    python scripts/verify_wheels.py wheelhouse --expect 2 --python 3.11
校验内容：
    1. 目录存在且至少包含指定数量的 .whl 文件
    2. 文件名符合 PEP 427 的 wheel 命名规范
    3. （可选）wheel 的 Python 标签与目标版本匹配
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
# PEP 427: {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
WHEEL_PATTERN = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"(-(?P<build>\d[^-]*))?"
    r"-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>.+)\.whl$"
)
def check_wheel_filename(path: Path) -> tuple[bool, str, dict[str, str]]:
    """校验单个 wheel 文件名，返回 (是否合法, 说明, 解析字段)。"""
    match = WHEEL_PATTERN.match(path.name)
    if not match:
        return False, f"文件名不符合 PEP 427 规范：{path.name}", {}
    return True, "命名规范校验通过", match.groupdict()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验 wheel 产物")
    parser.add_argument("wheelhouse", help="wheel 所在目录")
    parser.add_argument(
        "--expect",
        type=int,
        default=1,
        help="期望的最少 wheel 数量（默认 1）",
    )
    parser.add_argument(
        "--python",
        default="",
        help="目标 Python 版本，例如 3.11；填写后校验 Python 标签",
    )
    args = parser.parse_args(argv)
    house = Path(args.wheelhouse)
    if not house.is_dir():
        print(f"[FAIL] 目录不存在：{house}")
        return 1
    wheels = sorted(house.glob("*.whl"))
    print(f"[INFO] 目录 {house} 中找到 {len(wheels)} 个 wheel")
    if len(wheels) < args.expect:
        print(f"[FAIL] 期望至少 {args.expect} 个 wheel，实际 {len(wheels)} 个")
        return 1
    failed = False
    for wheel in wheels:
        ok, message, fields = check_wheel_filename(wheel)
        size_kb = wheel.stat().st_size / 1024
        if not ok:
            print(f"[FAIL] {wheel.name} -> {message}")
            failed = True
            continue
        print(
            f"[PASS] {wheel.name} "
            f"({size_kb:.1f} KB, py={fields['py']}, "
            f"abi={fields['abi']}, plat={fields['plat']})"
        )
        # 目标 Python 版本标签校验
        if args.python:
            tag = args.python.replace(".", "")
            py_tags = fields["py"].split(".")
            if not any(tag in t or t == "py3" for t in py_tags):
                print(
                    f"[WARN] {wheel.name} 的 Python 标签 {fields['py']} "
                    f"未包含目标版本 {args.python}（纯 Python 包可忽略）"
                )
    if failed:
        return 1
    print("\n===== wheel 校验全部通过 =====")
    return 0
if __name__ == "__main__":
    sys.exit(main())
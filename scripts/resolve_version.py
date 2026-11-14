"""查询 PyPI 上与目标 Python 版本兼容的最新包版本号，供 GitHub Actions 调用。
用法：
    python scripts/resolve_version.py xgboost --python 3.11
    python scripts/resolve_version.py scikit-learn --python 3.11 --fallback 1.5.2
逻辑：
    1. 拉取 PyPI JSON API 的全部发布版本
    2. 逐个检查每个版本的 requires_python 是否兼容目标 Python
    3. 在兼容版本中选最新的一个打印到 stdout
输出：仅打印版本号到 stdout（便于 $(...) 捕获）。
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version
    _HAVE_PACKAGING = True
except ImportError:  # pragma: no cover - CI 中 build 依赖会带上 packaging
    _HAVE_PACKAGING = False
def fetch_pypi_payload(package: str, timeout: int = 60) -> dict:
    """从 PyPI JSON API 获取包的完整元数据。"""
    url = f"https://pypi.org/pypi/{package}/json"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "build-wheels-workflow/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
def naive_version_key(version: str) -> tuple:
    """简易版本排序键：提取数字段，非数字段置 0。仅作 packaging 缺失时的兜底。"""
    parts = re.split(r"[.\-+]", version)
    key: list[int] = []
    for part in parts:
        digits = re.match(r"^(\d+)", part)
        key.append(int(digits.group(1)) if digits else 0)
    while len(key) < 5:
        key.append(0)
    return tuple(key)
def naive_requires_python_ok(spec: str, py: tuple[int, int]) -> bool:
    """简易 requires_python 判断，支持 >=, >, <=, <, ==, != 与 .* 后缀。
    仅在 packaging 库不可用时使用。
    """
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        wildcard = clause.endswith(".*")
        if wildcard:
            clause = clause[:-2]
        match = re.match(r"^(>=|<=|==|!=|>|<)?\s*(\d+(?:\.\d+)*)$", clause)
        if not match:
            return True  # 无法解析的子句不拦截，交给上层兜底
        op, ver = match.group(1) or ">=", match.group(2)
        target = tuple(int(x) for x in ver.split("."))
        probe = py + (0,) * (len(target) - len(py))
        if op == ">=" and not probe >= target:
            return False
        if op == ">" and not probe > target:
            return False
        if op == "<=" and not probe <= target:
            return False
        if op == "<" and not probe < target:
            return False
        if op == "==" and probe[: len(target)] != target:
            return False
        if op == "!=" and wildcard is False and probe[: len(target)] == target:
            return False
    return True
def is_compatible(requires_python: str | None, py_version: tuple[int, int]) -> bool:
    """判断 requires_python 约束是否兼容目标 Python 版本。"""
    if not requires_python:
        return True
    if _HAVE_PACKAGING:
        try:
            return f"{py_version[0]}.{py_version[1]}" in SpecifierSet(requires_python)
        except InvalidVersion:
            return True
    return naive_requires_python_ok(requires_python, py_version)
def resolve_compatible_version(
    payload: dict,
    py_version: tuple[int, int],
) -> tuple[str, list[str]]:
    """在全部发布版本中筛选兼容目标 Python 的最新版本。
    返回 (选中版本, 被跳过的不兼容版本列表)。
    """
    releases: dict = payload.get("releases") or {}
    compatible: list[str] = []
    skipped: list[str] = []
    for version, files in releases.items():
        if not files:
            # 纯占位版本（无任何分发文件），跳过
            skipped.append(version)
            continue
        # 同一版本的多个文件可能声明不同 requires_python，
        # 只要有一个文件兼容目标 Python 即视为可用
        any_ok = any(
            is_compatible(f.get("requires_python"), py_version) for f in files
        )
        if any_ok:
            compatible.append(version)
        else:
            skipped.append(version)
    if not compatible:
        raise RuntimeError(
            f"没有找到兼容 Python {py_version[0]}.{py_version[1]} 的发布版本"
        )
    if _HAVE_PACKAGING:
        try:
            best = max(compatible, key=Version)
        except InvalidVersion:
            best = max(compatible, key=naive_version_key)
    else:
        best = max(compatible, key=naive_version_key)
    return best, sorted(skipped)
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="查询 PyPI 包中兼容目标 Python 的最新版本"
    )
    parser.add_argument("package", help="包名，例如 xgboost")
    parser.add_argument(
        "--python",
        default="",
        help="目标 Python 版本，例如 3.11；留空则不过滤兼容性",
    )
    parser.add_argument(
        "--fallback",
        default="",
        help="查询失败时使用的兜底版本号",
    )
    args = parser.parse_args(argv)
    py_version: tuple[int, int] | None = None
    if args.python:
        match = re.match(r"^(\d+)\.(\d+)", args.python)
        if not match:
            print(f"[ERROR] 无法解析 Python 版本：{args.python}", file=sys.stderr)
            return 1
        py_version = (int(match.group(1)), int(match.group(2)))
    try:
        payload = fetch_pypi_payload(args.package)
        if py_version is None:
            version = payload.get("info", {}).get("version")
            if not version:
                raise RuntimeError(f"PyPI 返回中缺少版本信息：{args.package}")
        else:
            version, skipped = resolve_compatible_version(payload, py_version)
            if skipped:
                print(
                    f"[INFO] 已跳过 {len(skipped)} 个不兼容 "
                    f"Python {py_version[0]}.{py_version[1]} 的版本，"
                    f"其中最新的几个：{skipped[-5:]}",
                    file=sys.stderr,
                )
    except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
        if args.fallback:
            print(
                f"[WARN] 查询 {args.package} 版本失败（{exc}），"
                f"使用兜底版本 {args.fallback}",
                file=sys.stderr,
            )
            print(args.fallback)
            return 0
        print(f"[ERROR] 查询 {args.package} 版本失败：{exc}", file=sys.stderr)
        return 1
    print(version)
    return 0
if __name__ == "__main__":
    sys.exit(main())

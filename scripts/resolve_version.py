"""查询 PyPI 上指定包的最新版本号，供 GitHub Actions 调用。
用法：
    python scripts/resolve_version.py xgboost
    python scripts/resolve_version.py scikit-learn --fallback 1.5.2
输出：仅打印版本号到 stdout（便于 $(...) 捕获）。
"""
from __future__ import annotations
import argparse
import json
import sys
import urllib.error
import urllib.request
def fetch_latest_version(package: str, timeout: int = 60) -> str:
    """从 PyPI JSON API 获取包的最新稳定版本号。"""
    url = f"https://pypi.org/pypi/{package}/json"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "build-wheels-workflow/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    version = payload.get("info", {}).get("version")
    if not version:
        raise RuntimeError(f"PyPI 返回中缺少版本信息：{package}")
    return version
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="查询 PyPI 包最新版本")
    parser.add_argument("package", help="包名，例如 xgboost")
    parser.add_argument(
        "--fallback",
        default="",
        help="查询失败时使用的兜底版本号",
    )
    args = parser.parse_args(argv)
    try:
        version = fetch_latest_version(args.package)
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
"""构建后的冒烟测试：验证 scikit-learn 与 xgboost 可正常导入并训练。
在已安装 wheel 的环境中运行：
    python scripts/smoke_test.py
"""
from __future__ import annotations
import sys
def main() -> int:
    try:
        import sklearn
        import xgboost
    except ImportError as exc:
        print(f"[FAIL] 导入失败：{exc}")
        return 1
    print(f"[INFO] scikit-learn: {sklearn.__version__}")
    print(f"[INFO] xgboost     : {xgboost.__version__}")
    # 版本一致性：两者都要求 Python 3.11 环境
    if sys.version_info[:2] != (3, 11):
        print(
            f"[WARN] 当前 Python 为 {sys.version_info.major}."
            f"{sys.version_info.minor}，预期 3.11"
        )
    try:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=5,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        clf = XGBClassifier(
            n_estimators=10,
            max_depth=3,
            verbosity=0,
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(f"[INFO] 冒烟测试准确率: {accuracy:.4f}")
    except Exception as exc:  # noqa: BLE001 - 冒烟测试需要捕获全部异常
        print(f"[FAIL] 训练流程异常：{type(exc).__name__}: {exc}")
        return 1
    print("[PASS] SMOKE TEST PASSED")
    return 0
if __name__ == "__main__":
    sys.exit(main())
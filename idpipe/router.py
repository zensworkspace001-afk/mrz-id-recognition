"""路由器：MRZ 優先判斷簽發國；讀不出來才用影像分類器；兩者不一致則標記複核。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .templates import TEMPLATES, template_for_mrz

# 分類器介面：輸入影像，回傳 (template_key 或 'unknown', 信心)
Classifier = Callable[[object], tuple]

CLASSIFIER_MIN_CONF = 0.90


@dataclass
class RouteResult:
    template_key: str  # 'unknown' 表示走通用路徑
    method: str  # mrz | classifier | none
    reasons: list


def route(image, mrz_country: Optional[str], classifier: Optional[Classifier] = None,
          mrz_format: str = "TD3") -> RouteResult:
    reasons = []
    mrz_key = template_for_mrz(mrz_format, mrz_country) if mrz_country else None
    if mrz_country and not mrz_key:
        reasons.append(f"MRZ 簽發國 {mrz_country}（{mrz_format}）不在支援清單")

    cls_key, cls_conf = ("unknown", 0.0)
    if classifier is not None:
        cls_key, cls_conf = classifier(image)
        if cls_conf < CLASSIFIER_MIN_CONF:
            cls_key = "unknown"

    if mrz_key:
        if cls_key not in ("unknown", mrz_key):
            reasons.append(f"MRZ 判定 {mrz_key} 與分類器 {cls_key} 不一致")
        return RouteResult(mrz_key, "mrz", reasons)
    if cls_key in TEMPLATES:
        return RouteResult(cls_key, "classifier", reasons)
    reasons.append("無法判定版型，使用通用抽取")
    return RouteResult("unknown", "none", reasons)

"""隨機假身分。姓名池只是示意用的常見字詞，組合出來的人不對應任何真實個人。"""
from __future__ import annotations

import random
from datetime import date, timedelta

# (姓池, 名池)。越南姓名含變音符號、菲律賓含 Ñ，用來測試正規化。
NAME_POOLS = {
    "JPN": (["SATO", "SUZUKI", "TAKAHASHI", "TANAKA", "WATANABE"], ["HARUTO", "YUI", "SOTA", "AOI", "REN"]),
    "CHN": (["WANG", "LI", "ZHANG", "LIU", "CHEN"], ["WEI", "FANG", "NA", "MING", "JING"]),
    "HKG": (["CHAN", "WONG", "LEUNG", "LAM", "HO"], ["TAI MAN", "SIU MING", "KA YAN", "WING SZE", "CHI KIT"]),
    "KOR": (["KIM", "LEE", "PARK", "CHOI", "JUNG"], ["MIN JUN", "SEO YEON", "JI HOON", "HA EUN", "DO YUN"]),
    "VNM": (["NGUYỄN", "TRẦN", "LÊ", "PHẠM", "HOÀNG", "ĐẶNG"], ["VĂN AN", "THỊ HÀ", "MINH ĐỨC", "THANH TÙNG", "NGỌC ANH"]),
    "IDN": (["", "", ""], ["SUHARTO", "SITI", "BUDI", "WAYAN", "DEWI"]),  # 單名：姓為空
    "PHL": (["DELA CRUZ", "REYES", "SANTOS", "PEÑA", "IBAÑEZ"], ["MARIA CLARA", "JOSE", "ANA LIZA", "JUAN CARLO", "MARICEL"]),
    "IND": (["SHARMA", "PATEL", "SINGH", "KUMAR", "REDDY"], ["RAHUL", "PRIYA", "ANJALI", "ARJUN", "LAKSHMI"]),
}
SUPPORTED = list(NAME_POOLS)


def random_identity(country: str, rng: random.Random, today: date | None = None) -> dict:
    today = today or date(2026, 9, 25)
    surnames, givens = NAME_POOLS[country]
    surname = rng.choice(surnames)
    given = rng.choice(givens)
    mononym = surname == ""
    if mononym:
        surname, given = given, ""  # 單名放在 surname 欄（MRZ 慣例以實際樣本為準）
    dob = today - timedelta(days=rng.randint(18 * 365, 60 * 365))
    issue = today - timedelta(days=rng.randint(30, 8 * 365))
    expiry = issue + timedelta(days=10 * 365)
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    doc_no = rng.choice(letters) + "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([7, 8])))
    return {
        "country": country, "surname": surname, "given_names": given, "mononym": mononym,
        "sex": rng.choice(["M", "F"]), "dob": dob, "issue": issue, "expiry": expiry,
        "document_number": doc_no, "nationality": country,
    }

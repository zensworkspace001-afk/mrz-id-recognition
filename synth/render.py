"""合成證件渲染與拍攝條件擴增。

輸出標注格式與真實標注相同：四角點、條件標籤、欄位框（拉正後座標）與文字。
渲染出的圖帶有 SPECIMEN 浮水印，僅作模型訓練用途，不得當作證件使用。
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from idpipe.mrz import build_td3

def _font(*paths):
    """回傳第一個存在的字型路徑（Linux DejaVu 優先，macOS 備援）。"""
    return next((p for p in paths if Path(p).exists()), paths[0])


MONO = _font("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
             "/System/Library/Fonts/Supplemental/Andale Mono.ttf")
SANS = _font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/System/Library/Fonts/Supplemental/Arial.ttf")
PAGE_W, PAGE_H = 1250, 880  # 護照資料頁（ID-3 尺寸比例 125 x 88 mm）


def _fmt_date(d, style="DD MMM YYYY"):
    mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"][d.month - 1]
    return f"{d.day:02d} {mon} {d.year}" if style == "DD MMM YYYY" else d.strftime("%d/%m/%Y")


def render_passport_page(ident: dict, date_style="DD MMM YYYY"):
    """回傳 (PIL 影像, 標注 dict)。版面是通用示意版型，不是任何國家的真實版型。"""
    img = Image.new("RGB", (PAGE_W, PAGE_H), (238, 232, 214))
    d = ImageDraw.Draw(img)
    f_lab = ImageFont.truetype(SANS, 18)
    f_val = ImageFont.truetype(MONO, 34)
    f_mrz = ImageFont.truetype(MONO, 40)
    fields = []

    def put(name, label, text, x, y):
        d.text((x, y), label, fill=(90, 90, 90), font=f_lab)
        d.text((x, y + 24), text, fill=(20, 20, 20), font=f_val)
        w = d.textlength(text, font=f_val)
        fields.append({"name": name, "text": text, "legible": True,
                       "polygon": [[x, y + 24], [x + w, y + 24], [x + w, y + 68], [x, y + 68]]})

    put("surname", "Surname", ident["surname"], 380, 90)
    put("given_names", "Given names", ident["given_names"] or "-", 380, 180)
    put("nationality", "Nationality", ident["nationality"], 380, 270)
    put("date_of_birth", "Date of birth", _fmt_date(ident["dob"], date_style), 380, 360)
    put("sex", "Sex", ident["sex"], 800, 360)
    put("document_number", "Passport No.", ident["document_number"], 380, 450)
    put("date_of_issue", "Date of issue", _fmt_date(ident["issue"], date_style), 380, 540)
    put("date_of_expiry", "Date of expiry", _fmt_date(ident["expiry"], date_style), 800, 540)
    d.rectangle([60, 90, 320, 400], outline=(150, 150, 150), width=3)  # 照片框

    l1, l2 = build_td3(ident["country"], ident["surname"], ident["given_names"], ident["document_number"],
                       ident["nationality"], ident["dob"].strftime("%y%m%d"), ident["sex"],
                       ident["expiry"].strftime("%y%m%d"))
    cw = d.textlength("A", font=f_mrz)
    x0 = (PAGE_W - cw * 44) / 2
    for i, line in enumerate((l1, l2)):
        y = 720 + i * 60
        d.text((x0, y), line, fill=(15, 15, 15), font=f_mrz)
        fields.append({"name": f"mrz_line{i + 1}", "text": line, "legible": True,
                       "polygon": [[x0, y], [x0 + cw * 44, y], [x0 + cw * 44, y + 50], [x0, y + 50]]})

    # SPECIMEN 浮水印
    wm = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(wm).text((250, 300), "SPECIMEN", fill=(200, 0, 0, 70), font=ImageFont.truetype(SANS, 170))
    img = Image.alpha_composite(img.convert("RGBA"), wm).convert("RGB")
    return img, {"mrz": [l1, l2], "fields": fields}


def _background(w, h, rng):
    base = np.array([rng.randint(40, 200) for _ in range(3)], dtype=np.uint8)
    bg = np.tile(base, (h, w, 1)).astype(np.float32)
    bg += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(4, 25), bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def augment(page: Image.Image, rng: random.Random, out_size=(1280, 960)):
    """套用透視、旋轉、光線、陰影、反光、模糊、雜訊。回傳 (拍攝影像 BGR, 四角點, 條件標籤, 拉正用的 H)。"""
    W, H = out_size
    src = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    h0, w0 = src.shape[:2]
    scale = rng.uniform(0.45, 0.85) * min(W / w0, H / h0)
    cx, cy = W / 2 + rng.uniform(-0.08, 0.08) * W, H / 2 + rng.uniform(-0.08, 0.08) * H
    ang = np.deg2rad(rng.uniform(-25, 25))
    tilt = rng.uniform(0, 0.25)  # 透視強度
    corners = np.array([[0, 0], [w0, 0], [w0, h0], [0, h0]], dtype=np.float32) - [w0 / 2, h0 / 2]
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (corners @ R.T) * scale
    ax = rng.choice([0, 1])
    sgn = rng.choice([-1, 1])
    for i, (x, y) in enumerate(dst):  # 簡單的梯形畸變
        k = 1 + sgn * tilt * (y / (h0 * scale) if ax == 0 else x / (w0 * scale))
        dst[i] = [x * k, y] if ax == 0 else [x, y * k]
    dst += [cx, cy]
    dst = dst.astype(np.float32)
    M = cv2.getPerspectiveTransform(np.array([[0, 0], [w0, 0], [w0, h0], [0, h0]], dtype=np.float32), dst)

    bg = _background(W, H, rng)
    warped = cv2.warpPerspective(src, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h0, w0), 255, np.uint8), M, (W, H))
    img = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)

    cond = {}
    # 亮度與對比
    gain = rng.uniform(0.45, 1.25)
    cond["light"] = "dim" if gain < 0.7 else ("bright" if gain > 1.1 else "normal")
    img = img * gain + rng.uniform(-20, 20)
    # 漸層陰影
    if rng.random() < 0.5:
        gx = np.linspace(rng.uniform(0.5, 1), rng.uniform(0.9, 1.1), W)[None, :, None]
        img = img * gx
        cond["shadow"] = True
    else:
        cond["shadow"] = False
    # 反光
    cond["glare"] = rng.random() < 0.35
    if cond["glare"]:
        mx, my = dst.mean(axis=0)  # 反光落在證件範圍內
        gcx, gcy = int(mx + rng.uniform(-0.25, 0.25) * W * 0.5), int(my + rng.uniform(-0.25, 0.25) * H * 0.5)
        gr = rng.randint(80, 260)
        yy, xx = np.ogrid[:H, :W]
        g = np.exp(-(((xx - gcx) ** 2 + (yy - gcy) ** 2) / (2 * (gr / 2) ** 2)))
        img = img + g[..., None] * rng.uniform(80, 220)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # 模糊與雜訊
    blur = rng.choice([0, 0, 3, 5, 7])
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    cond["blur"] = blur
    noise = rng.uniform(0, 8)
    img = np.clip(img + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 92)])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    cond["angle_deg"] = round(float(np.rad2deg(ang)), 1)
    cond["tilt"] = round(tilt, 3)
    return img, dst.tolist(), cond, M


def generate_sample(country: str, idx: int, out_dir: Path, seed: int) -> dict:
    from .identities import random_identity
    rng = random.Random(f"{seed}-{country}-{idx}")  # 含國家，避免各國同編號樣本套用完全相同的拍攝條件
    ident = random_identity(country, rng)
    style = "DD/MM/YYYY" if country in ("VNM", "IND") else "DD MMM YYYY"
    page, ann = render_passport_page(ident, style)
    photo, corners, cond, M = augment(page, rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{country}_{idx:05d}"
    path = out_dir / f"{name}.jpg"
    cv2.imwrite(str(path), photo)
    return {
        "image": path.name, "image_id": name, "doc_type": "passport", "country": country,
        "template_key": f"{country}-passport-current", "corners": corners, "conditions": cond,
        "fields": ann["fields"], "mrz": ann["mrz"], "synthetic": True,
        "note": "版面為通用示意版型，非任何國家真實版型；SPECIMEN 浮水印",
    }

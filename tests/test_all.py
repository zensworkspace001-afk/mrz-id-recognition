import csv
import os
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idpipe.mrz import build_td1, build_td3, check_digit, find_mrz_lines, parse_mrz, parse_td1, parse_td3
from idpipe.normalize import names_match, parse_date, strip_diacritics, taiwan_id_valid
from idpipe.pipeline import Components, Pipeline
from idpipe.router import route
from idpipe.storage import JsonlStorage, SqliteStorage, export_csv

ICAO_L1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
ICAO_L2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


class TestMrz(unittest.TestCase):
    def test_icao_sample(self):
        r = parse_td3(ICAO_L1, ICAO_L2)
        self.assertTrue(r.checksum_valid)
        self.assertEqual((r.surname, r.given_names), ("ERIKSSON", "ANNA MARIA"))
        self.assertEqual(r.date_of_birth, "1974-08-12")

    def test_check_digit_tamper(self):
        bad = ICAO_L2[:5] + "9" + ICAO_L2[6:]
        r = parse_td3(ICAO_L1, bad)
        self.assertFalse(r.checksum_valid)
        self.assertFalse(r.checks["document_number"])

    def test_build_roundtrip_all_countries(self):
        from synth.identities import SUPPORTED, random_identity
        rng = random.Random(1)
        for c in SUPPORTED:
            for _ in range(50):
                i = random_identity(c, rng)
                l1, l2 = build_td3(c, i["surname"], i["given_names"], i["document_number"], c,
                                   i["dob"].strftime("%y%m%d"), i["sex"], i["expiry"].strftime("%y%m%d"))
                r = parse_td3(l1, l2)
                self.assertTrue(r.checksum_valid, (c, l1, l2))
                self.assertEqual(r.date_of_expiry, i["expiry"].isoformat())

    def test_find_mrz_with_ocr_confusion(self):
        # 生日欄（純數字）的 0 被讀成 O、1 被讀成 I：依位置修正
        noisy = ICAO_L2.replace("7408122", "74O8I22")
        l1, l2 = find_mrz_lines("PASSPORT\nSome text\n" + ICAO_L1 + "\n" + noisy)
        self.assertEqual(l2, ICAO_L2)
        # 護照號碼欄字母數字皆可：靠檢查碼判斷 O/0 混淆
        noisy = ICAO_L2.replace("L898902C3", "L8989O2C3")
        l1, l2 = find_mrz_lines(ICAO_L1 + "\n" + noisy)
        self.assertEqual(l2, ICAO_L2)

    def test_find_mrz_with_dropped_fillers(self):
        # Tesseract 實際輸出：連續的 '<' 被漏掉，兩行長度都不是 44
        text = "P<CHNLI<<NA<<<<\nG1238360<7CHN9111288F3503258<<<<<<<<4\n"
        l1, l2 = find_mrz_lines(text)
        r = parse_td3(l1, l2)
        self.assertTrue(r.checksum_valid)
        self.assertEqual((r.issuing_country, r.surname, r.given_names), ("CHN", "LI", "NA"))
        self.assertEqual(r.date_of_expiry, "2035-03-25")

    def test_name_digits_fixed(self):
        text = "P<INDPATEL<<LAKSHM1<<<<<<<<<<<<<<<<<<<<<<<<<\n" + ICAO_L2
        l1, _ = find_mrz_lines(text)
        self.assertIn("LAKSHMI", l1)

    def test_country_misread_reconciled(self):
        # 實際 OCR 輸出：國籍 JPN 讀成 3PN、簽發國 IND 讀成 TND；生日 6 讀成 G
        l1, l2 = build_td3("JPN", "TAKAHASHI", "HARUTO", "J13858530", "JPN", "051015", "M", "310111")
        a, b = find_mrz_lines(l1 + "\n" + l2.replace("JPN", "3PN"))
        self.assertEqual((a, b), (l1, l2))
        l1, l2 = build_td3("IND", "KUMAR", "PRIYA", "Z1234567", "IND", "680921", "F", "300101")
        a, b = find_mrz_lines(l1.replace("IND", "TND") + "\n" + l2.replace("680921", "G80921"))
        self.assertEqual((a, b), (l1, l2))

    def test_first_line_second_char_misread(self):
        # 實拍護照：P<TWN… 的 '<' 被讀成 S
        l1, l2 = build_td3("TWN", "LIANG", "MEI HUA", "300000001", "TWN", "900131", "F", "310130", "A123456789")
        a, b = find_mrz_lines("PS" + l1[2:] + "\n" + l2)
        self.assertEqual((a, b), (l1, l2))

    def test_mononym(self):
        l1, l2 = build_td3("IDN", "SUHARTO", "", "A1234567", "IDN", "800101", "M", "300101")
        r = parse_td3(l1, l2)
        self.assertEqual((r.surname, r.given_names), ("SUHARTO", ""))


# 移民署樣張背面 MRZ。居留證樣張第一、二行經 OCR 與目視一致；金卡第二行 < 的位置看不清楚。
# 兩張的總檢查碼都與 ICAO 公式不符（居留證印 8、算得 3），需以真實卡片確認，因此這裡不驗證總檢查碼。
GOLD_CARD = ("I<TWN12345678<84YZ000000080983", "9001011F2101314JPN0072<<<<<<<0", "LIN<MAY<LEE" + "<" * 19)
ARC_NEW = ("I<TWNP2000050170A9000000160100", "0005016F2105013<<<0152<<<<<<<8", "THIRTY<DAYS" + "<" * 19)


class TestTd1(unittest.TestCase):
    def test_official_samples(self):
        # 證件號碼欄是護照號碼；統一證號在選用資料欄；姓名行不以 << 分姓名
        r = parse_td1(*GOLD_CARD)
        self.assertTrue(all(r.checks[k] for k in ("document_number", "date_of_birth", "date_of_expiry")))
        self.assertEqual((r.doc_code, r.issuing_country, r.document_number, r.nationality), ("I", "TWN", "12345678", "JPN"))
        self.assertIn("YZ00000008", "".join(r.optional_data))
        self.assertEqual((r.surname, r.given_names), ("LIN MAY LEE", ""))
        r = parse_td1(*ARC_NEW)
        self.assertTrue(all(r.checks[k] for k in ("document_number", "date_of_birth", "date_of_expiry")))
        self.assertEqual((r.document_number, r.date_of_birth, r.date_of_expiry, r.sex), ("P20000501", "2000-05-01", "2021-05-01", "F"))
        self.assertIn("A900000016", "".join(r.optional_data))

    def test_build_roundtrip_and_long_doc_number(self):
        l1, l2, l3 = build_td1("TWN", "NGUYEN VAN AN", "C12345678", "VNM", "900131", "M", "300101", "0A9123456780001")
        r = parse_td1(l1, l2, l3)
        self.assertTrue(r.checksum_valid, (l1, l2, l3))
        # 證件號碼超過 9 碼：檢查碼位置為 <，其餘接在選用資料欄
        l1 = "I<UTOD23145890<7349<<<<<<<<<<<"
        self.assertEqual(parse_td1(l1, "3407127M9507122UTO<<<<<<<<<<<2", "STEVENSON<<PETER<JOHN<<<<<<<<<").document_number,
                         "D23145890734")

    def test_find_td1_with_ocr_noise(self):
        l1, l2, l3 = build_td1("TWN", "LIN MAY LEE", "12345678", "JPN", "900101", "F", "210131", "4YZ000000080983")
        noisy = "\n".join(["NATIONAL IMMIGRATION AGENCY", l1, l2.replace("900101", "9OO1O1").replace("<<<<<", "<<"), l3.rstrip("<")])
        lines = find_mrz_lines(noisy)
        self.assertEqual(lines, (l1, l2, l3))
        self.assertTrue(parse_mrz(lines).checksum_valid)


class TestDetect(unittest.TestCase):
    def test_reject_non_document_shapes(self):
        from idpipe.detect import plausible_document
        self.assertTrue(plausible_document([[0, 0], [1250, 0], [1250, 880], [0, 880]]))  # 護照 1.42
        self.assertTrue(plausible_document([[0, 0], [856, 0], [856, 540], [0, 540]]))  # 卡片 1.59
        # 實際案例：已裁切的居留證照片，框到了中間文字區塊（長寬比約 2.35）
        self.assertFalse(plausible_document([[0, 78], [283, 91], [329, 213], [0, 213]]))

    def test_shadow_gradient_and_out_of_frame(self):
        import cv2
        import numpy as np
        from idpipe.detect import detect_corners
        from idpipe.rectify import order_corners
        h, w = 600, 800
        # 背景由左到右明暗漸層（陰影）；證件右下角超出畫面
        gx = np.linspace(40, 170, w, dtype=np.float32)[None, :, None]
        img = np.broadcast_to(gx * np.array([1.0, 0.8, 1.2], np.float32), (h, w, 3)).astype(np.uint8).copy()
        truth = np.array([[180, 120], [760, 90], [840, 520], [220, 560]], np.float32)  # (840,520) 在畫面外
        cv2.fillPoly(img, [truth.astype(np.int32)], (225, 230, 235))
        c = detect_corners(img)
        self.assertIsNotNone(c)
        err = np.linalg.norm(np.array(order_corners(c)) - np.array(order_corners(truth.tolist())), axis=1)
        self.assertLess(err.max(), 6, c)

    def test_locate_mrz_any_rotation(self):
        import cv2
        import numpy as np
        from synth.identities import random_identity
        from synth.render import render_passport_page
        from idpipe.locate import locate_mrz
        page, _ = render_passport_page(random_identity("JPN", random.Random(5)))
        page = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        canvas = np.full((1800, 1800, 3), (60, 90, 130), np.uint8)
        canvas[460:1340, 275:1525] = page
        for deg, rotate in ((0, None), (90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180)):
            img = canvas if rotate is None else cv2.rotate(canvas, rotate)
            loc = locate_mrz(img)
            self.assertIsNotNone(loc, deg)
            self.assertEqual(loc.format, "TD3")
            # 推算的頁面（正反兩個方向之一）應與實際頁面四角相近
            truth = np.array([[275, 460], [1525, 460], [1525, 1340], [275, 1340]], np.float64)
            if rotate is not None:
                M = {cv2.ROTATE_90_CLOCKWISE: lambda p: [1799 - p[1], p[0]], cv2.ROTATE_180: lambda p: [1799 - p[0], 1799 - p[1]]}[rotate]
                truth = np.array([M(p) for p in truth])
            errs = [np.abs(np.sort(np.array(loc.page_quad(f)), 0) - np.sort(truth, 0)).max() for f in (False, True)]
            self.assertLess(min(errs), 90, (deg, errs))  # 合成版面與 ICAO 比例略有不同，容許約 7% 誤差

    def test_locate_ignores_page_edges(self):
        # 實拍：護照疊起來的頁緣是兩條等長、平行、比 MRZ 更長的線，曾被誤認為 MRZ
        import cv2
        import numpy as np
        from synth.identities import random_identity
        from synth.render import render_passport_page
        from idpipe.locate import locate_mrz
        page, _ = render_passport_page(random_identity("KOR", random.Random(2)))
        img = np.full((1300, 1500, 3), 235, np.uint8)
        img[100:980, 125:1375] = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        for y in (1010, 1045):
            cv2.line(img, (60, y), (1440, y), (70, 70, 70), 9)
        loc = locate_mrz(img)
        self.assertIsNotNone(loc)
        ys = sorted(float(np.mean(np.array(l)[:, 1])) for l in loc.lines)
        self.assertTrue(all(800 < y < 960 for y in ys), ys)  # MRZ 在頁面 y≈820～930（原圖座標）

    def test_photo_quality_check(self):
        import cv2
        import numpy as np
        from idpipe.quality import check_photo
        q = check_photo(cv2.imread("data/synth/images/JPN_00001.jpg"))
        self.assertTrue(q["usable"], q["checks"])
        q = check_photo(np.full((800, 1200, 3), 200, np.uint8))
        self.assertFalse(q["usable"])
        self.assertEqual(q["checks"][0]["key"], "mrz")
        # MRZ 大面積過曝 -> 不適用
        q = check_photo(cv2.imread("data/synth/images/PHL_00007.jpg"))
        self.assertIn(("glare", "fail"), [(c["key"], c["status"]) for c in q["checks"]])

    def test_cropped_card_uses_full_frame(self):
        import numpy as np
        from idpipe.detect import detect_corners
        # 證件佔滿整張圖（沒有背景）：淺色卡面、中下方一塊深色文字區
        img = np.full((214, 330, 3), (240, 232, 214), np.uint8)
        img[80:200, 10:320] = (60, 60, 60)
        img[0:25, 30:200] = (30, 30, 30)
        c = detect_corners(img)
        self.assertIn(c, (None, [[0.0, 0.0], [329.0, 0.0], [329.0, 213.0], [0.0, 213.0]]))


class TestChevron(unittest.TestCase):
    def _glyph(self, ch):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from synth.render import MONO
        img = Image.new("L", (60, 80), 255)
        ImageDraw.Draw(img).text((8, 5), ch, fill=0, font=ImageFont.truetype(MONO, 60))
        return np.array(img) < 128

    def test_chevron_shape(self):
        from idpipe.chevron import is_chevron
        self.assertTrue(is_chevron(self._glyph("<")))
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":  # 4 的形狀最接近 '<'（曾誤判）
            self.assertFalse(is_chevron(self._glyph(ch)), ch)

    def test_align_ocr_to_glyph_slots(self):
        from idpipe.chevron import align
        truth = "P<TWNLIANG<<MEI<HUA" + "<" * 25  # 虛構姓名，錯誤型態取自實拍護照
        slots = [c == "<" for c in truth]
        # 實拍護照的 OCR 錯誤型態：'<' 讀成 S/K、多讀一個 L、行尾多兩個雜字
        ocr = "P<TWNLILANG<<MEISHUAS<" + "K" * 22 + "SS"
        fixed = align(slots, ocr)
        self.assertEqual(len(fixed), 44)
        self.assertEqual(fixed[19:], "<" * 25)
        self.assertEqual([i for i, c in enumerate(fixed) if c == "<"], [i for i, c in enumerate(truth) if c == "<"])


def _line(text, x0, y0, x1, y1, score=0.95):
    import numpy as np
    from idpipe.ppocr import TextLine
    return TextLine(text, score, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32))


class TestVizLabels(unittest.TestCase):
    """標籤定位解析（不需要模型）：文字行是依實拍台灣護照的 OCR 結果構造的，姓名與號碼為虛構。"""

    def _page(self, sex=True):
        L = [_line("中華民國REPUBLIC OF CHINA", 368, 114, 900, 134), _line("PASSPORT", 29, 148, 140, 166),
             _line("通知人 (IN CASE OF EMERGENCY, NOTIFY THE INDIVIDUAL NAMED BELOW)", 100, 20, 900, 40),
             _line("護照號碼 / Passport No.", 732, 208, 940, 224), _line("300000001", 737, 230, 900, 246),
             _line("姓名/ Nme (Sairabe, Giveu mames)", 375, 260, 700, 276),
             _line("王美華", 372, 288, 450, 306), _line("WANG, MEI-HUA", 520, 289, 700, 305),
             _line("國籍/Nationality", 374, 330, 520, 345), _line("REPUBLIC OF CHINA", 376, 352, 600, 368),
             _line("身分證统一編號 / Personal Id. No.", 740, 330, 1000, 346), _line("A123456789", 741, 354, 900, 370),
             _line("性别/Sex", 373, 386, 460, 401),
             _line("出生日期/Date of birth", 739, 386, 950, 401), _line("15 MAY 2003", 743, 411, 900, 427),
             _line("發照日期/ Date of issuic", 376, 445, 600, 461), _line("01SEP2014", 379, 470, 520, 486),
             _line("出生地 / Place of birth", 740, 443, 950, 458), _line("TAIWAN", 744, 469, 850, 485),
             _line("效期截止日期/ Date of expiry", 374, 502, 640, 518), _line("01 SEP 2019", 377, 527, 520, 543),
             _line("發照機關/Authority", 378, 561, 560, 577), _line("MINISTRYOF FOREIGN AFFAIRS", 375, 586, 780, 602)]
        if sex:
            L.append(_line("M", 378, 414, 392, 430))
        return L

    def _extract(self, lines):
        from unittest import mock
        from idpipe import vizlabels
        with mock.patch.object(vizlabels, "engine", return_value=lambda img: lines):
            import numpy as np
            return vizlabels.extract(np.zeros((880, 1250, 3), np.uint8),
                                     {"document_number": r"\d{9}", "personal_number": r"[A-Z][12]\d{8}"})

    def test_fields_from_labels(self):
        v = {k: x["raw"] for k, x in self._extract(self._page()).items()}
        self.assertEqual((v["name_native"], v["surname"], v["given_names"]), ("王美華", "WANG", "MEI-HUA"))
        self.assertEqual((v["document_number"], v["personal_number"], v["sex"]), ("300000001", "A123456789", "M"))
        self.assertEqual((v["date_of_birth"], v["date_of_issue"], v["date_of_expiry"]), ("15 MAY 2003", "01 SEP 2014", "01 SEP 2019"))
        self.assertEqual((v["place_of_birth"], v["nationality"]), ("TAIWAN", "REPUBLIC OF CHINA"))

    def test_missing_value_does_not_steal_next_field(self):
        v = self._extract(self._page(sex=False))
        self.assertNotIn("sex", v)  # 性別的值沒偵測到時，不能抓到下方發照日期的值

    def test_format_fallback_when_label_unreadable(self):
        lines = [l for l in self._page() if "Personal" not in l.text]
        self.assertEqual(self._extract(lines)["personal_number"]["raw"], "A123456789")

    def test_ppocr_reads_rendered_text(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime 未安裝")
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from synth.render import MONO
        from idpipe.ppocr import engine
        img = Image.new("RGB", (900, 200), (240, 238, 230))
        d = ImageDraw.Draw(img)
        d.text((40, 40), "MINISTRY OF FOREIGN AFFAIRS", fill=(20, 20, 20), font=ImageFont.truetype(MONO, 36))
        d.text((40, 120), "15 MAY 2003", fill=(20, 20, 20), font=ImageFont.truetype(MONO, 36))
        texts = [l.text.replace(" ", "") for l in engine()(np.array(img)[:, :, ::-1].copy())]
        self.assertIn("MINISTRYOFFOREIGNAFFAIRS", texts)
        self.assertIn("15MAY2003", texts)


class TestLicense(unittest.TestCase):
    @unittest.skipIf(os.environ.get("IDPIPE_ALLOW_PIP_OPENCV") == "1",
                     "雲端部署（例如 Vercel）刻意使用 pip 的 opencv-python-headless：只在伺服器執行、不散布，見 THIRD_PARTY_NOTICES.md")
    def test_opencv_has_no_video_backends(self):
        # pip 的 opencv-python* 內附 FFmpeg（含 GPL 的 x264/x265）；本專案要用 scripts/build_opencv.sh 編的版本
        import cv2
        info = cv2.getBuildInformation()
        for backend in ("FFMPEG", "GStreamer", "AVFoundation"):
            self.assertNotRegex(info, rf"{backend}:\s+YES", f"OpenCV 啟用了 {backend}；請改用 scripts/build_opencv.sh")
        self.assertFalse(hasattr(cv2, "VideoCapture"), "OpenCV 含 videoio 模組；請改用 scripts/build_opencv.sh")


class TestNormalize(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(parse_date("31 JAN 1990"), "1990-01-31")
        self.assertEqual(parse_date("31 JAN/JAN 1990"), "1990-01-31")
        self.assertEqual(parse_date("31/01/1990"), "1990-01-31")
        self.assertEqual(parse_date("1990.01.31"), "1990-01-31")
        self.assertEqual(parse_date("1990年1月31日"), "1990-01-31")
        self.assertIsNone(parse_date("31/02/1990"))

    def test_taiwan_id(self):
        self.assertTrue(taiwan_id_valid("A123456789"))
        self.assertFalse(taiwan_id_valid("A123456788"))
        self.assertTrue(taiwan_id_valid("A900000016"))  # 新式統一證號（移民署居留證樣張）
        self.assertFalse(taiwan_id_valid("YZ00000008"))  # 舊式居留證號不適用此演算法

    def test_names(self):
        self.assertEqual(strip_diacritics("NGUYỄN ĐỨC"), "NGUYEN DUC")
        self.assertTrue(names_match("NGUYỄN", "NGUYEN"))
        self.assertTrue(names_match("PEÑA", "PENA"))
        self.assertFalse(names_match("LEE", "KIM"))


class TestRouter(unittest.TestCase):
    def test_mrz_first(self):
        self.assertEqual(route(None, "VNM").template_key, "VNM-passport-current")

    def test_conflict_flagged(self):
        r = route(None, "VNM", lambda img: ("JPN-passport-current", 0.99))
        self.assertEqual(r.template_key, "VNM-passport-current")
        self.assertTrue(r.reasons)

    def test_classifier_fallback_and_unknown(self):
        self.assertEqual(route(None, None, lambda i: ("KOR-passport-current", 0.95)).method, "classifier")
        self.assertEqual(route(None, None, lambda i: ("KOR-passport-current", 0.5)).template_key, "unknown")
        self.assertEqual(route(None, "USA").template_key, "unknown")


class TestPipeline(unittest.TestCase):
    def _pipe(self, ocr_text, viz=None, storage=None):
        return Pipeline(Components(ocr_text=lambda i: ocr_text, viz_extract=lambda i, t: viz or {}), storage)

    def test_clean_passport(self):
        l1, l2 = build_td3("VNM", "NGUYỄN", "VĂN AN", "B1234567", "VNM", "900131", "M", "310130")
        viz = {"surname": {"raw": "NGUYỄN", "confidence": 0.97},
               "date_of_expiry": {"raw": "30/01/2031", "confidence": 0.95},
               "place_of_birth": {"raw": "HA NOI", "confidence": 0.9}}
        rec = self._pipe(l1 + "\n" + l2, viz).process(None, b"img1")
        self.assertEqual(rec.template_key, "VNM-passport-current")
        self.assertEqual(rec.date_of_expiry, "2031-01-30")
        self.assertEqual(rec.place_of_birth, "HA NOI")
        self.assertFalse(rec.needs_review, rec.review_reasons)

    def test_mismatch_flags_review(self):
        l1, l2 = build_td3("JPN", "SATO", "HARUTO", "TK1234567", "JPN", "900131", "M", "310130")
        viz = {"date_of_expiry": {"raw": "31 JAN 2031", "confidence": 0.95}}
        rec = self._pipe(l1 + "\n" + l2, viz).process(None, b"img2")
        self.assertTrue(rec.needs_review)
        self.assertIn("date_of_expiry 視覺區與 MRZ 不一致", rec.review_reasons)

    def test_bad_checksum_and_low_conf(self):
        l1, l2 = build_td3("KOR", "KIM", "MIN JUN", "M1234567", "KOR", "900131", "F", "310130")
        l2 = l2[:5] + ("9" if l2[5] != "9" else "8") + l2[6:]
        rec = self._pipe(l1 + "\n" + l2, {"place_of_birth": {"raw": "SEOUL", "confidence": 0.4}}).process(None, b"img3")
        self.assertTrue(any("檢查碼" in r for r in rec.review_reasons))
        self.assertTrue(any("信心低" in r for r in rec.review_reasons))

    def test_viz_name_crosscheck(self):
        # MRZ 第一行沒有檢查碼：靠視覺區姓名攔下 OCR 多讀的字母；沒截斷時不接受前綴相符
        l1, l2 = build_td3("JPN", "SATO", "HARUTO", "TK1234567", "JPN", "900131", "M", "310130")
        l1 = l1.replace("HARUTO<", "HARUTOO")
        viz = {"surname": {"raw": "SATO", "confidence": 0.95}, "given_names": {"raw": "HARUTO", "confidence": 0.95}}
        rec = self._pipe(l1 + "\n" + l2, viz).process(None, b"img7")
        self.assertEqual(rec.review_reasons, ["given_names 視覺區與 MRZ 不一致"])

    def test_low_conf_viz_defers_to_valid_mrz(self):
        # MRZ 檢查碼通過時，低信心的視覺區讀值不觸發不一致，也不拉低 MRZ 欄位信心
        l1, l2 = build_td3("KOR", "KIM", "MIN JUN", "M1234567", "KOR", "900131", "F", "310130")
        viz = {"date_of_birth": {"raw": "81 JAN 1990", "confidence": 0.3}}
        rec = self._pipe(l1 + "\n" + l2, viz).process(None, b"img8")
        self.assertFalse(rec.needs_review, rec.review_reasons)
        self.assertEqual(rec.fields["date_of_birth"]["confidence"], 1.0)

    def test_name_noise_flagged(self):
        l1, l2 = build_td3("IDN", "WAYAN", "", "W7910271", "IDN", "721007", "F", "310703")
        l1 = l1[:20] + "CEE<KEEEE" + l1[29:]
        rec = self._pipe(l1 + "\n" + l2).process(None, b"img6")
        self.assertIn("MRZ 姓名欄有雜訊", rec.review_reasons)

    def test_arc_with_td1_mrz(self):
        l1, l2, l3 = build_td1("TWN", "LIN MAY LEE", "12345678", "JPN", "900101", "F", "310131", "4YZ000000080983")
        viz = {"surname": {"raw": "LIN", "confidence": 0.95}, "given_names": {"raw": "MAY LEE", "confidence": 0.95},
               "arc_number": {"raw": "YZ00000008", "confidence": 0.95}}
        rec = self._pipe("\n".join((l1, l2, l3)), viz).process(None, b"img9")
        self.assertEqual((rec.template_key, rec.doc_type), ("TWN-arc-current", "arc"))
        self.assertEqual((rec.residence.arc_number, rec.residence.valid_until, rec.document_number),
                         ("YZ00000008", "2031-01-31", "12345678"))
        self.assertFalse(rec.needs_review, rec.review_reasons)
        # 視覺區統一證號與 MRZ 不同 -> 複核；姓名以全名比對
        viz["arc_number"]["raw"] = "YZ00000009"
        viz["given_names"]["raw"] = "MAY LEF"
        rec = self._pipe("\n".join((l1, l2, l3)), viz).process(None, b"img10")
        self.assertIn("arc_number 視覺區與 MRZ 不一致", rec.review_reasons)
        self.assertIn("姓名視覺區與 MRZ 不一致", rec.review_reasons)

    def test_taiwan_passport_personal_id(self):
        l1, l2 = build_td3("TWN", "WANG", "HSIAO-MING", "300000001", "TWN", "900131", "M", "310130", "A123456789")
        rec = self._pipe(l1 + "\n" + l2).process(None, b"img11")
        self.assertEqual((rec.template_key, rec.personal_number), ("TWN-passport-current", "A123456789"))
        self.assertFalse(rec.needs_review, rec.review_reasons)
        # 身分證字號檢查碼錯（MRZ 個人號碼檢查碼照樣重算成通過）-> 複核
        l1, l2 = build_td3("TWN", "WANG", "HSIAO-MING", "300000001", "TWN", "900131", "M", "310130", "A123456788")
        rec = self._pipe(l1 + "\n" + l2).process(None, b"img12")
        self.assertIn("身分證統一編號格式或檢查碼不符", rec.review_reasons)

    def test_taiwan_passport_viz_crosscheck(self):
        l1, l2 = build_td3("TWN", "WANG", "MEI HUA", "300000001", "TWN", "030515", "F", "190901", "A223456781")
        viz = {"surname": {"raw": "WANG", "confidence": 0.99}, "given_names": {"raw": "MEI-HUA", "confidence": 0.99},
               "personal_number": {"raw": "A223456781", "confidence": 0.99},
               "date_of_issue": {"raw": "01 SEP 2014", "confidence": 0.98}, "name_native": {"raw": "王美華", "confidence": 0.9}}
        rec = self._pipe(l1 + "\n" + l2, viz).process(None, b"img13")
        self.assertFalse(rec.needs_review, rec.review_reasons)
        self.assertEqual((rec.personal_number, rec.name_native), ("A223456781", "王美華"))  # MRZ 為準，視覺區不覆蓋
        viz["personal_number"]["raw"] = "A223456782"
        viz["date_of_issue"]["raw"] = "02 SEP 2014"  # 台灣護照效期 5 或 10 年，差一天不合理
        rec = self._pipe(l1 + "\n" + l2, viz).process(None, b"img14")
        self.assertIn("personal_number 視覺區與 MRZ 不一致", rec.review_reasons)
        self.assertTrue(any(x.startswith("發照日期與出生日期或到期日不符") for x in rec.review_reasons), rec.review_reasons)
        self.assertEqual(rec.personal_number, "A223456781")

    def test_no_mrz_unknown(self):
        rec = self._pipe("nothing here").process(None, b"img4")
        self.assertEqual(rec.template_key, "unknown")
        self.assertTrue(rec.needs_review)

    def test_arc(self):
        p = Pipeline(Components(classifier=lambda i: ("TWN-arc-current", 0.97),
                                viz_extract=lambda i, t: {
                                    "arc_number": {"raw": "AB12345678", "confidence": 0.95},
                                    "valid_until": {"raw": "2027/03/15", "confidence": 0.93},
                                    "employer": {"raw": "TEST CO LTD", "confidence": 0.9}}))
        rec = p.process(None, b"img5")
        self.assertEqual(rec.doc_type, "arc")
        self.assertEqual(rec.residence.valid_until, "2027-03-15")
        self.assertEqual(rec.residence.employer, "TEST CO LTD")
        self.assertFalse(rec.needs_review, rec.review_reasons)
        bad = Pipeline(Components(classifier=lambda i: ("TWN-arc-current", 0.97),
                                  viz_extract=lambda i, t: {"arc_number": {"raw": "12", "confidence": 0.95}}))
        self.assertIn("居留證號格式不符", bad.process(None, b"img6").review_reasons)


class TestStorage(unittest.TestCase):
    def test_jsonl_sqlite_csv(self):
        l1, l2 = build_td3("IND", "SHARMA", "RAHUL", "K0012345", "IND", "900131", "M", "310130")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            js, sq = JsonlStorage(d / "r.jsonl"), SqliteStorage(d / "r.db")
            for st in (js, sq):
                Pipeline(Components(ocr_text=lambda i: l1 + "\n" + l2), st).process(None, b"same")
                Pipeline(Components(ocr_text=lambda i: l1 + "\n" + l2), st).process(None, b"same")  # 重複
                self.assertEqual(len(list(st.all())), 1)  # 以 doc_id 去重
            n = export_csv(js.all(), d / "out.csv")
            self.assertEqual(n, 1)
            with (d / "out.csv").open(encoding="utf-8-sig") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["issuing_country"], "IND")
            self.assertEqual(row["document_number"].strip(), "K0012345")
            # 與 SQLite 內容一致
            self.assertEqual(js.get(row["doc_id"]), sq.get(row["doc_id"]))


class TestSqlExport(unittest.TestCase):
    def _rows(self):
        from idpipe.sqlexport import row_of
        l1, l2 = build_td3("TWN", "O'NEIL", "MEI-LING", "300000001", "TWN", "900131", "F", "310130", "A123456789")
        rec = Pipeline(Components(ocr_text=lambda i: l1 + "\n" + l2)).process(None, b"sql1").to_dict()
        rec["name_native"] = "王美玲"
        return [row_of(rec, "a.jpg")], rec

    def test_sqlite_create_then_upsert(self):
        import sqlite3
        from idpipe.sqlexport import row_of, to_sql
        rows, rec = self._rows()
        db = sqlite3.connect(":memory:")
        db.executescript(to_sql(rows, "sqlite", "docs"))  # 沒有資料表 -> 建立並寫入
        rec["review_reasons"] = ["人工修正"]
        db.executescript(to_sql([row_of(rec, "a.jpg", reviewed=True)], "sqlite", "docs"))  # 已存在 -> 更新同一筆
        self.assertEqual(db.execute("select count(*) from docs").fetchone()[0], 1)
        got = db.execute("select given_names, name_native, personal_number, reviewed_by_human, review_reasons from docs").fetchone()
        self.assertEqual(got, ("MEI LING", "王美玲", "A123456789", 1, "人工修正"))

    def test_dialects_and_table_name(self):
        from idpipe.sqlexport import to_sql
        rows, _ = self._rows()
        self.assertIn("ON CONFLICT", to_sql(rows, "postgresql"))
        self.assertIn("ON DUPLICATE KEY UPDATE", to_sql(rows, "mysql"))
        self.assertIn("MERGE [id_documents]", to_sql(rows, "sqlserver"))
        self.assertIn("N'王美玲'", to_sql(rows, "sqlserver"))
        with self.assertRaises(ValueError):
            to_sql(rows, "postgresql", "x; DROP TABLE y")


class TestSynthEndToEnd(unittest.TestCase):
    def test_rectify_with_annotated_corners(self):
        import cv2
        from idpipe.rectify import order_corners, rectify
        from synth.render import generate_sample
        with tempfile.TemporaryDirectory() as d:
            ann = generate_sample("VNM", 7, Path(d), seed=3)
            img = cv2.imread(str(Path(d) / ann["image"]))
            self.assertEqual(len(ann["corners"]), 4)
            self.assertEqual(order_corners(ann["corners"][::-1]), order_corners(ann["corners"]))
            flat = rectify(img, ann["corners"])
            self.assertEqual(flat.shape[:2], (880, 1250))
            # MRZ 標注文字必須能通過檢查碼
            r = parse_td3(*ann["mrz"])
            self.assertTrue(r.checksum_valid)
            self.assertEqual(r.issuing_country, "VNM")

    def test_viz_extract_synth_layout(self):
        import cv2
        from idpipe.rectify import rectify
        from idpipe.templates import SYNTH_LAYOUT, TEMPLATES
        from idpipe.viz import make_viz_extractor
        from synth.render import generate_sample
        with tempfile.TemporaryDirectory() as d:
            ann = generate_sample("PHL", 3, Path(d), seed=1)
            flat = rectify(cv2.imread(str(Path(d) / ann["image"])), ann["corners"])
            viz = make_viz_extractor(SYNTH_LAYOUT)(flat, TEMPLATES[ann["template_key"]])
            truth = {f["name"]: f["text"] for f in ann["fields"]}
            self.assertTrue(names_match(viz["surname"]["raw"], truth["surname"], allow_prefix=False))
            self.assertEqual(parse_date(viz["date_of_birth"]["raw"]), parse_date(truth["date_of_birth"]))


if __name__ == "__main__":
    unittest.main(verbosity=1)

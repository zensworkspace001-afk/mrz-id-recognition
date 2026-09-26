# 第三方元件授權聲明（THIRD_PARTY_NOTICES）

本專案使用下列第三方元件。各元件的授權原文放在 `licenses/`。
清單依實際安裝與編譯結果整理（2026-09-25，macOS arm64）；更新版本或換平台後請重新核對。

**結論**：本機版（`requirements-local.txt` + `scripts/build_opencv.sh`）執行期使用的元件全部是寬鬆授權（Apache-2.0、BSD、MIT 類），**不含 GPL／LGPL 元件**。
雲端展示版（Vercel，`requirements.txt`）另見第 6 節：含 GPL／LGPL 元件，但只在伺服器上執行、不散布。
對外散布時須附上本檔與 `licenses/`，並保留下方「必要的著作權聲明」。

本檔為工程整理，不構成法律意見；正式散布或商用前請由法務確認。

## 1. 執行期元件

### Python 套件（`requirements-local.txt`）

| 元件 | 版本 | 授權 | 用途 | 授權原文 |
|---|---|---|---|---|
| NumPy | 2.5.3 | BSD-3-Clause（內附元件另含 0BSD、MIT、Zlib 等寬鬆授權） | 陣列運算 | `numpy-LICENSE.txt`、`numpy-bundled/` |
| pytesseract | 0.3.13 | Apache-2.0 | 呼叫 Tesseract | `pytesseract-LICENSE.txt` |
| Pillow | 12.3.0 | MIT-CMU（HPND） | pytesseract 相依；合成資料繪圖 | `pillow-LICENSE.txt`（含下列內附函式庫全文） |
| packaging | 26.3 | Apache-2.0 或 BSD-2-Clause | pytesseract 相依 | `packaging-LICENSE.*.txt` |
| onnxruntime | 1.30.0 | MIT（內附元件見其 ThirdPartyNotices） | 視覺區 PP-OCR 模型推論 | `onnxruntime-LICENSE.txt`、`onnxruntime-ThirdPartyNotices.txt` |
| protobuf | 7.36.2 | BSD-3-Clause | onnxruntime 相依 | `protobuf-LICENSE.txt` |
| flatbuffers | 25.12.19 | Apache-2.0 | onnxruntime 相依 | `opencv-3rdparty-flatbuffers.txt`（同為 Apache-2.0 全文） |

### PP-OCR 模型（`models/ppocr/`）

PaddleOCR 的 PP-OCRv6 small 文字偵測與辨識模型（Apache-2.0，© PaddlePaddle Authors），ONNX 檔取自 RapidOCR 3.9.2 的發行檔（Apache-2.0）。
`idpipe/ppocr.py` 依 RapidOCR 的預設參數重寫前後處理，**不安裝 rapidocr 套件**——它會連帶安裝 pip 版 `opencv-python`（含 GPL 的 FFmpeg）與 `shapely`（內含 LGPL 的 GEOS）。
授權全文與出處：`ppocr-models-Apache-2.0.txt`（同一份也放在 `models/ppocr/LICENSE`）。

Pillow 的 wheel 內附原生函式庫：libjpeg-turbo（IJG／BSD-3／Zlib）、libpng（libpng）、libtiff（libtiff）、libwebp（BSD-3）、zlib（Zlib）、FreeType（FTL，與 GPLv2 雙授權，**本專案選用 FTL**）、HarfBuzz（MIT）、Little CMS（MIT）、OpenJPEG（BSD-2）、Brotli（MIT）、liblzma（0BSD）、libavif（BSD-2）、libxcb／libXau（MIT），全文見 `pillow-LICENSE.txt`。

### OpenCV（`scripts/build_opencv.sh` 自行編譯）

| 元件 | 版本 | 授權 | 授權原文 |
|---|---|---|---|
| OpenCV（core、imgproc、imgcodecs、flann、geometry、python3） | 5.0.0 | Apache-2.0 | `opencv-LICENSE.txt` |
| zlib（內附） | 1.3.2 | Zlib | `opencv-3rdparty-zlib.txt` |
| libjpeg-turbo（內附） | 3.1.2 | IJG + BSD-3-Clause + Zlib | `opencv-3rdparty-libjpeg-turbo.md`、`…README.ijg` |
| libpng（內附） | 1.6.57 | libpng License v2 | `opencv-3rdparty-libpng.txt` |
| libtiff（內附） | 4.7.1 | libtiff（BSD 類） | `opencv-3rdparty-libtiff.md` |
| libwebp（內附） | 1.6.0 | BSD-3-Clause | `opencv-3rdparty-libwebp.txt` |
| FlatBuffers（內附） | 25.9.23 | Apache-2.0 | `opencv-3rdparty-flatbuffers.txt` |
| carotene HAL（ARM 加速） | 0.0.1 | BSD-3-Clause（NVIDIA） | `opencv-hal-carotene-BSD-3.txt` |
| KleidiCV HAL（ARM 加速） | 26.03 | Apache-2.0（Arm） | `opencv-3rdparty-kleidicv-Apache-2.0.txt` |

編譯時關閉：FFmpeg、GStreamer、AVFoundation 等所有影片功能（videoio 模組整個不編）、GUI（Qt／GTK）、
OpenEXR、AVIF、JPEG 2000、JPEG XL、Intel IPP、OpenCL、Eigen、LAPACK、Protobuf。
影像格式函式庫全部由 OpenCV 內附原始碼靜態編入，不依賴系統上的其他版本。

### 系統程式（Homebrew 安裝，不隨 Python 套件打包）

| 元件 | 版本 | 授權 | 授權原文 |
|---|---|---|---|
| Tesseract OCR（含 eng／osd 語言資料） | 5.5.3 | Apache-2.0 | `tesseract-LICENSE.txt` |
| 英文語言模型 `tessdata/eng.traineddata`（隨 repo 附上，雲端版使用） | tessdata_fast | Apache-2.0 | `tessdata-eng-Apache-2.0.txt` |
| Leptonica | 1.87.0 | BSD-2-Clause | `leptonica-LICENSE.txt` |
| libarchive | 3.8.9 | BSD-2-Clause | `libarchive-COPYING.txt` |

`tesseract` 執行檔實際連結的只有上列三者、macOS 內建的 libcurl 與 Accelerate。
Homebrew 另外安裝的 cairo、pango、glib 等（LGPL）只供 Tesseract 的訓練工具使用，本專案沒有呼叫，
散布時也不需要附上；若改在其他平台部署，請重新確認連結的函式庫。

## 2. 必要的著作權聲明

對外散布含上述元件的軟體時，需在文件中保留：

- 本軟體部分功能基於 Independent JPEG Group 的成果（This software is based in part on the work of the Independent JPEG Group）。——libjpeg-turbo
- Portions of this software are copyright © The FreeType Project (www.freetype.org). All rights reserved. ——Pillow 內附 FreeType
- Apache-2.0 元件（OpenCV、Tesseract、pytesseract、KleidiCV、FlatBuffers、packaging、PP-OCR 模型）：附上授權全文；若該元件帶有 NOTICE 檔，須一併保留其內容。

## 3. 刻意排除的元件

| 元件 | 排除原因 |
|---|---|
| pip 套件 `opencv-python`、`opencv-python-headless` | 內附 FFmpeg，含 **GPL** 的 libx264、libx265、librubberband、libvidstab，且 H.264／H.265 編碼受專利池涵蓋。本專案只處理靜態影像，用不到。 |

`tests/test_all.py` 的 `TestLicense` 會在測試時檢查目前載入的 OpenCV 有沒有啟用 FFmpeg 或其他影片後端，若有人改回 pip 版本，測試會失敗。

## 4. 只在開發時使用（不隨軟體散布）

- CMake（BSD-3-Clause）、Xcode Command Line Tools：編譯 OpenCV 用。
- 合成資料字型：Linux 用 DejaVu（Bitstream Vera／Public Domain 類授權）；macOS 備援用系統內建的 Andale Mono、Arial。系統字型只用來產生合成影像，字型檔本身不複製、不散布。

## 5. 資料

- `data/synth/`：本專案自行產生的合成資料，身分全為隨機捏造，版面為通用示意版型，帶 SPECIMEN 浮水印。
- 官方證件樣張（例如移民署公告的居留證樣張）只在內部測試時參考，未收錄於本專案，也不應放入要散布的產品或資料集。

## 6. 雲端展示版（Vercel）

Vercel 無法安裝系統程式、也無法執行 `scripts/build_opencv.sh`，所以雲端版改用 pip 上的現成套件（`requirements.txt`）：

| 元件 | 版本 | 授權 | 說明 |
|---|---|---|---|
| tesserocr | 2.11.0 | MIT（`tesserocr-LICENSE.txt`） | Linux wheel 內含 libtesseract 5.5.1（Apache-2.0）、Leptonica（BSD-2）、libjpeg／libpng／libtiff／libwebp（寬鬆授權），以及 **libjbig（jbigkit，GPL-2.0）** |
| cysignals | 1.12.6 | **LGPL-3.0-or-later** | tesserocr 的相依套件 |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 | wheel 內含 FFmpeg，其中 **libx264／libx265／librubberband 為 GPL** |
| numpy、pillow、onnxruntime（與 PP-OCR 模型） | 同上 | 同第 1 節 | |

這些元件只在 Vercel 的伺服器上執行，使用者透過網頁使用服務、不會取得程式本身，**不構成散布**，GPL／LGPL 的散布義務不會觸發
（GPL 與 LGPL 不含 AGPL 那種「透過網路提供服務也要公開原始碼」的條款）。
本 repo 只列出相依套件名稱，不收錄這些 wheel 的二進位檔。

若要把軟體交給客戶或其他公司部署（屬於散布），請改用本機版的組合，或在 Linux 上以 `scripts/build_opencv.sh`
編譯精簡 OpenCV、並以原始碼編譯不含 libjbig 的 Tesseract 取代上述 wheel。
`tests/test_all.py` 的 `TestLicense` 在雲端環境可設 `IDPIPE_ALLOW_PIP_OPENCV=1` 跳過。

## 更新本檔

升級套件或換平台時：

1. `.venv/bin/pip list` 與 `brew info tesseract leptonica` 核對版本。
2. 看各 wheel 內的 `.dylibs/`（Linux 為 `.libs/`）有沒有新增原生函式庫。
3. 重新執行 `scripts/build_opencv.sh`，確認最後一行顯示 `video backends enabled: none`。
4. 執行 `python -m unittest discover -s tests`（含 `TestLicense`）。

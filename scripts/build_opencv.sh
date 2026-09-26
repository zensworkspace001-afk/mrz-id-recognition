#!/usr/bin/env bash
# 從原始碼編譯「不含 FFmpeg／影片功能」的 OpenCV Python 模組，安裝到指定的 Python 環境。
#
# 為什麼不用 pip 的 opencv-python / opencv-python-headless：
#   這兩個 wheel 都打包了 FFmpeg，內含 GPL 授權的 libx264、libx265、librubberband，
#   以及受 H.264/H.265 專利池涵蓋的編碼器。本專案只處理靜態影像，完全用不到影片功能。
#
# 只編 core、imgproc、imgcodecs 三個模組（本專案用到的全部 cv2 函式都在這裡），
# 影像格式只保留 JPEG / PNG / WebP / TIFF，且由 OpenCV 內附的原始碼靜態編入（不依賴系統函式庫）。
# 結果：單一 cv2 模組，所含第三方元件皆為寬鬆授權（見 THIRD_PARTY_NOTICES.md）。
#
# 用法：scripts/build_opencv.sh [python 執行檔，預設 .venv/bin/python] [工作目錄，預設 build/opencv]
# 需要：cmake、C++ 編譯器（macOS：Xcode Command Line Tools）、目標 Python 環境已裝 numpy。
set -euo pipefail

OPENCV_VERSION=5.0.0
PY=${1:-.venv/bin/python}
WORK=${2:-build/opencv}
PY=$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')
NUMPY_INC=$("$PY" -c 'import numpy; print(numpy.get_include())')
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

mkdir -p "$WORK"
cd "$WORK"
if [ ! -d "opencv-$OPENCV_VERSION" ]; then
  curl -sSL -o "opencv-$OPENCV_VERSION.tar.gz" "https://github.com/opencv/opencv/archive/refs/tags/$OPENCV_VERSION.tar.gz"
  tar xzf "opencv-$OPENCV_VERSION.tar.gz"
fi

cmake -S "opencv-$OPENCV_VERSION" -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LIST=core,imgproc,imgcodecs,python3 \
  -DBUILD_SHARED_LIBS=OFF \
  -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_AVFOUNDATION=OFF -DVIDEOIO_ENABLE_PLUGINS=OFF \
  -DWITH_OPENEXR=OFF -DWITH_AVIF=OFF -DWITH_JASPER=OFF -DWITH_OPENJPEG=OFF -DWITH_GDAL=OFF -DWITH_GDCM=OFF \
  -DWITH_JPEGXL=OFF -DWITH_SPNG=OFF -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_SUNRASTER=OFF \
  -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_PFM=OFF \
  -DBUILD_ZLIB=ON -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_WEBP=ON -DBUILD_TIFF=ON \
  -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF -DWITH_PROTOBUF=OFF \
  -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_WIN32UI=OFF -DWITH_1394=OFF -DWITH_V4L=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_DOCS=OFF \
  -DBUILD_JAVA=OFF -DBUILD_opencv_python2=OFF -DOPENCV_SKIP_PYTHON_LOADER=ON \
  -DPYTHON3_EXECUTABLE="$PY" -DPYTHON3_NUMPY_INCLUDE_DIRS="$NUMPY_INC" \
  -DOPENCV_PYTHON3_INSTALL_PATH="$SITE" -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build -j "$JOBS"
cmake --install build
if [ "$(uname)" = Darwin ]; then  # 安裝時改寫過路徑，重新做 ad-hoc 簽章（Apple Silicon 載入需要）
  codesign --force -s - "$SITE"/cv2*.so
fi

"$PY" - <<'EOF'
import re, cv2
info = cv2.getBuildInformation()
bad = [b for b in ("FFMPEG", "GStreamer", "AVFoundation") if re.search(rf"{b}:\s+YES", info)]
print("OpenCV", cv2.__version__, "installed at", cv2.__file__)
print("video backends enabled:", bad or "none")
EOF

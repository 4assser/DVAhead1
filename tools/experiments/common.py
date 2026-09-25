"""Общие пути и утилиты для экспериментов."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GLB = Path(os.environ.get('DVA_GLB', REPO / 'DvaGolovaOK_2_lite.glb'))
OUT = Path(os.environ.get('DVA_OUT', HERE / 'out'))          # промежуточные файлы (в .gitignore)
IMG = Path(os.environ.get('DVA_IMG', REPO / 'docs' / 'img'))  # итоговые картинки для отчёта
GNM_DIR = Path(os.environ.get('GNM_DIR', REPO.parent / 'GNM'))  # git clone https://github.com/google/GNM
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def font(size=18):
    from PIL import ImageFont
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/Library/Fonts/Arial Unicode.ttf',
              'C:/Windows/Fonts/arial.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

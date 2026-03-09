import numpy as np
import pandas as pd
import cv2
from PIL import Image
from wordcloud import WordCloud
import random

# ===== 路径改这里 =====
KITTY_IMG = r"neg.jpg"                      # 你的kitty图片（用于提轮廓mask）
FONT = r"C:\Windows\Fonts\msyh.ttc"         # 中文字体
OUT_PNG = r"wordcloud_mouse_shape.png"      # 输出词云

# 词频表：二选一
POS_XLSX = r"pos_top_words.xlsx"            # 列名 Word/Count
NEG_XLSX = r"neg_top_words_new.xlsx"        # 列名 词/频次
USE_POS = False                              # True用正向；False用负向

import random
def blue_palette_color_func(*args, **kwargs):
    # hue: 200~220 大致是蓝色区间（偏天蓝到深蓝）
    h = random.randint(200, 220)
    s = random.randint(55, 90)   # 饱和度
    l = random.randint(35, 70)   # 亮度（越大越浅蓝）
    return f"hsl({h}, {s}%, {l}%)"


def load_freq():
    if USE_POS:
        df = pd.read_excel(POS_XLSX)
        if "Word" not in df.columns or "Count" not in df.columns:
            raise ValueError(f"{POS_XLSX} 需要列名: Word / Count，当前列名={list(df.columns)}")
        return dict(zip(df["Word"].astype(str), df["Count"].astype(int)))
    else:
        df = pd.read_excel(NEG_XLSX)
        if "词" not in df.columns or "频次" not in df.columns:
            raise ValueError(f"{NEG_XLSX} 需要列名: 词 / 频次，当前列名={list(df.columns)}")
        return dict(zip(df["词"].astype(str), df["频次"].astype(int)))


def build_filled_silhouette_mask(img_path, out_debug="mask_debug.png"):
    """从图片中提取最大轮廓并填充为实心mask：0可放词，255不可放词"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # 放大一点更平滑
    h, w = img.shape[:2]
    scale = 3 if max(h, w) < 600 else 2
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # 灰度 + 模糊
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 二值化：把主体从白背景中分离（反色）
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 闭运算：填洞、连通线条（减少“眼睛/胡须挖洞”）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 找最大外轮廓
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found. Try using a clearer silhouette image.")

    largest = max(contours, key=cv2.contourArea)

    # 填充轮廓成实心mask
    mask_fill = np.ones_like(bw, dtype=np.uint8) * 255  # 255=不放词
    cv2.drawContours(mask_fill, [largest], -1, color=0, thickness=-1)  # 0=可放词

    # 轻微圆润
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_fill = cv2.morphologyEx(mask_fill, cv2.MORPH_CLOSE, kernel2, iterations=1)

    # 保存调试mask
    Image.fromarray(mask_fill).save(out_debug)
    return mask_fill


def main():
    freq = load_freq()
    mask = build_filled_silhouette_mask(KITTY_IMG, out_debug="mask_debug.png")

    wc = WordCloud(
        font_path=FONT,
        background_color="white",      # 想透明底：改 None 并加 mode="RGBA"
        mask=mask,
        max_words=350,
        collocations=False,
        prefer_horizontal=0.9,
        color_func=blue_palette_color_func,  # ✅ 蓝色系字体
    ).generate_from_frequencies(freq)

    wc.to_file(OUT_PNG)
    print("Saved:", OUT_PNG)
    print("Mask debug saved: mask_debug.png (用来确认轮廓是否正确)")


if __name__ == "__main__":
    main()
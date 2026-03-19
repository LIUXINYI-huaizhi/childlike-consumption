import io
import time
import uuid
import random
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pymysql
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud

# =========================================================
# 基础设置
# =========================================================
st.set_page_config(
    page_title="测测你的童趣消费人格",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
FONT_DIR = BASE / "fonts"
IMG_DIR = BASE / "images"
CUT_DIR = BASE / "images_cut"
CUT_DIR.mkdir(exist_ok=True)

POS_XLSX = DATA_DIR / "pos_top_words.xlsx"
NEG_XLSX = DATA_DIR / "neg_top_words_new.xlsx"
MINIPROGRAM_QR = IMG_DIR / "miniprogram_code.png"

FONT_PATH = str(FONT_DIR / "msyh.ttc")
if not Path(FONT_PATH).exists():
    FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"

# =========================================================
# 样式
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.2rem;
        max-width: 1150px;
    }
    header[data-testid="stHeader"] {
        background: rgba(255,255,255,0);
        height: 0px;
    }
    div[data-testid="stToolbar"] { visibility: hidden; height: 0px; }

    html, body, [class*="css"]  {
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI",
                     "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei",
                     Arial, sans-serif;
    }

    .nav-wrap {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(255,255,255,.92);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 20px;
        padding: 10px 14px;
        box-shadow: 0 10px 30px rgba(2,6,23,.06);
        margin-bottom: 18px;
    }

    .hero-wrap { text-align: center; margin-top: 0.6rem; margin-bottom: 1.1rem; }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 900;
        line-height: 1.15;
        letter-spacing: .5px;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.0rem 0 0.25rem 0;
    }
    .hero-sub {
        color: #64748b;
        font-size: 1.05rem;
        margin: 0 auto;
        max-width: 820px;
    }

    .card {
        border: 1px solid rgba(148,163,184,.35);
        border-radius: 18px;
        padding: 16px 16px;
        background: rgba(255,255,255,.82);
        box-shadow: 0 8px 24px rgba(2,6,23,.08);
    }
    .muted { color: #64748b; }

    .pill {
        display:inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(37,99,235,.12);
        color: #1d4ed8;
        font-weight: 800;
        margin-top: 6px;
        margin-bottom: 10px;
    }
    .pill2 {
        display:inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(236,72,153,.12);
        color: #be185d;
        font-weight: 800;
        margin-top: 8px;
    }

    .mini-card{
        background: rgba(255,255,255,.85);
        border: 1px solid rgba(148,163,184,.28);
        border-radius: 18px;
        padding: 14px 14px;
        box-shadow: 0 10px 22px rgba(2,6,23,.06);
        margin-bottom: 12px;
    }

    .preview-card{
        border: 1px solid rgba(148,163,184,.28);
        border-radius: 18px;
        padding: 14px;
        background: linear-gradient(180deg, rgba(99,102,241,.04), rgba(236,72,153,.04));
        box-shadow: 0 10px 22px rgba(2,6,23,.05);
        height: 100%;
    }

    .profile-wrap{
        background: linear-gradient(180deg, rgba(99,102,241,.06), rgba(236,72,153,.05));
        border: 1px solid rgba(148,163,184,.35);
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 18px 40px rgba(2,6,23,.10);
    }
    .profile-head{
        display:flex; align-items:flex-end; justify-content:space-between;
        gap: 12px; margin-bottom: 14px;
    }
    .profile-title{
        font-size: 1.75rem; font-weight: 950; letter-spacing:.3px;
        margin: 0;
    }
    .profile-sub{
        color:#64748b; font-size:.95rem; margin: 4px 0 0 0;
    }
    .badge{
        display:inline-flex; align-items:center; gap:8px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(37,99,235,.12);
        color: #1d4ed8;
        font-weight: 900;
        font-size: .92rem;
        border: 1px solid rgba(37,99,235,.18);
    }

    .chips{ display:flex; flex-wrap:wrap; gap:8px; margin-top: 6px; }
    .chip{
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(236,72,153,.10);
        color:#be185d;
        font-weight: 800;
        font-size: .90rem;
        border: 1px solid rgba(236,72,153,.18);
    }

    .kv{
        display:grid;
        grid-template-columns: 92px 1fr;
        gap: 8px 10px;
        align-items: baseline;
        font-size: .98rem;
        line-height: 1.55;
    }
    .k{ color:#64748b; font-weight:700; }
    .v{ color:#0f172a; font-weight:650; }

    .list{ margin: 0; padding-left: 1rem; }
    .list li{ margin: .35rem 0; line-height: 1.55; font-size: .98rem; }

    div.stButton>button {
        border-radius: 14px;
        padding: 0.72rem 1rem;
        font-weight: 800;
        border: 1px solid rgba(148,163,184,.45);
        box-shadow: 0 10px 26px rgba(2,6,23,.08);
    }
    .stImage img{
        border-radius: 16px;
        box-shadow: 0 16px 40px rgba(2,6,23,.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 数据字典
# =========================================================
QUESTIONS = [
    ("压力大时，这类产品能让我放松或治愈", "A"),
    ("我会因为怀旧或童年情感而购买", "A"),
    ("我会因为同好/圈子氛围而更想买", "A"),

    ("拥有它们会让我感到愉悦或兴奋", "B"),
    ("把玩/摆放能让我更沉浸、更解压", "B"),
    ("我更在意“值不值、体验好不好”", "B"),

    ("我会为了分享、晒图、社交话题而买", "C"),
    ("抽选/惊喜感会明显提高我的兴趣", "C"),
    ("拿到喜欢款会想被别人看见/认可", "C"),

    ("我会很在意做工、质量和细节", "D"),
    ("我更容易被外观设计和颜值吸引", "D"),
    ("我更偏好稳定耐看、少踩雷", "D"),
]

TYPE_DESC = {
    "社交孔雀": "你更看重分享与话题感，喜欢新鲜感和“开箱的那一下”。",
    "圈层海王": "你更在意同好与归属感，喜欢围绕IP与系列收集建立连接。",
    "佛系水豚": "你更偏悦己与治愈，重视手感、颜值、品质与长期陪伴感。",
    "理智过客": "你整体更谨慎，倾向低风险试水或以送礼为主。"
}

TYPE_PREVIEW_WORDS = {
    "社交孔雀": ["晒图", "联名", "话题", "开箱", "惊喜"],
    "圈层海王": ["IP", "圈层", "谷子", "同好", "收集"],
    "佛系水豚": ["治愈", "颜值", "手感", "陪伴", "桌搭"],
    "理智过客": ["试水", "送礼", "稳妥", "口碑", "性价比"],
}

TYPE_IMG = {
    "社交孔雀": IMG_DIR / "social_peacock.jpg",
    "圈层海王": IMG_DIR / "circle_sea.jpg",
    "佛系水豚": IMG_DIR / "chill_cap.jpg",
    "理智过客": IMG_DIR / "rational_guest.jpg",
}

RECO = {
    "社交孔雀": {
        "tribe": "或许这里有你的同好：盲盒玩家、开箱党、打卡分享党",
        "brands_primary": ["泡泡玛特（POP MART）", "TOP TOY（名创优品旗下）", "52TOYS"],
        "channels": ["官方旗舰店/小程序", "线下门店与快闪", "正规电商自营（减少黄牛风险）"],
        "items": [
            {
                "title": "盲盒与联名限定",
                "types": ["盲盒", "联名限定", "快闪款"],
                "why": "话题感强，容易获得社交互动与惊喜体验。",
                "trigger_words": ["抽到", "联名", "喜欢", "可爱", "特别", "这次"],
                "risk_words": ["概率", "重复", "黄牛", "炒价", "溢价", "不值"],
            },
            {
                "title": "线下打卡与开箱分享向",
                "types": ["门店限定", "活动周边", "展会款"],
                "why": "更适合拍照分享与现场体验，强化社交氛围。",
                "trigger_words": ["喜欢", "可爱", "联名", "分享", "开箱", "这次"],
                "risk_words": ["排队", "断货", "黄牛", "溢价", "不值"],
            },
        ],
    },
    "圈层海王": {
        "tribe": "或许这里有你的同好：谷圈、娃圈、IP收藏党、同好社群",
        "brands_primary": ["泡泡玛特（联名线）", "名创优品联名", "官方IP店/旗舰店"],
        "channels": ["官方授权渠道（减少盗版）", "同好社群团购需谨慎", "线下IP主题店"],
        "items": [
            {
                "title": "IP周边与系列收集",
                "types": ["谷子（徽章/挂件）", "IP周边", "系列收集"],
                "why": "适合围绕同一个IP长期收集，强化归属感与圈层连接。",
                "trigger_words": ["喜欢", "联名", "特别", "这次", "可爱"],
                "risk_words": ["溢价", "不值", "质量", "做工", "退货"],
            },
            {
                "title": "棉花娃娃与“陪伴感”玩法",
                "types": ["棉花娃娃", "娃衣/配件", "桌搭周边"],
                "why": "更偏长期陪伴与社群互动，适合稳定投入。",
                "trigger_words": ["可爱", "手感", "柔软", "治愈", "喜欢"],
                "risk_words": ["掉毛", "做工", "异味", "色差", "退货"],
            },
        ],
    },
    "佛系水豚": {
        "tribe": "或许这里有你的同好：桌搭党、治愈系收集党、手感控",
        "brands_primary": ["52TOYS（摆件线）", "泡泡玛特（治愈系系列）", "TOP TOY（桌搭类）"],
        "channels": ["看口碑评价与细节图", "优先可退换渠道", "避免不明来源高溢价款"],
        "items": [
            {
                "title": "治愈桌搭小物（悦己向）",
                "types": ["摆件", "毛绒/小玩具", "桌搭周边"],
                "why": "强调颜值与手感，适合自我治愈与长期摆放。",
                "trigger_words": ["治愈", "解压", "手感", "柔软", "舒服", "喜欢"],
                "risk_words": ["做工", "质量", "瑕疵", "掉毛", "异味"],
            },
            {
                "title": "品质细节优先（少而精）",
                "types": ["手办/摆件", "高评价系列", "稳定供给款"],
                "why": "更适合追求稳定体验，降低踩雷概率。",
                "trigger_words": ["做工", "质量", "满意", "不错", "特别"],
                "risk_words": ["瑕疵", "破损", "色差", "退货", "售后"],
            },
        ],
    },
    "理智过客": {
        "tribe": "或许这里有你的同好：送礼党、轻度体验党、路人收藏",
        "brands_primary": ["经典摆件/手办", "解压捏捏", "毛绒挂件/钥匙扣", "IP联名文具与生活小物", "小盲盒"],
        "channels": ["优先可退换", "避免高溢价冲动买", "送礼先确认对方偏好"],
        "items": [
            {
                "title": "基础款试水（低风险）",
                "types": ["基础款", "经典款", "小件试水"],
                "why": "先小成本验证是否真的喜欢，再考虑升级。",
                "trigger_words": ["喜欢", "可爱", "不错", "满意"],
                "risk_words": ["不值", "溢价", "退货", "客服", "质量"],
            },
            {
                "title": "送礼向（更稳）",
                "types": ["礼盒", "联名小件", "轻量周边"],
                "why": "礼赠更看重“稳妥与好看”，不建议押注概率玩法。",
                "trigger_words": ["可爱", "特别", "喜欢", "这次"],
                "risk_words": ["溢价", "不值", "做工", "质量", "售后"],
            },
        ],
    },
}

PERSONA_PROFILE = {
    "社交孔雀": {
        "title": "社交打卡型用户",
        "traits": ["爱分享、爱开箱", "追求新鲜感与话题", "容易被联名/限定刺激"],
        "pay_items": ["盲盒/抽选", "联名限定周边", "线下快闪体验"],
        "price": "100–500元/月（热度高时可能更高）",
    },
    "圈层海王": {
        "title": "圈层归属型用户",
        "traits": ["围绕IP长期收集", "重视同好与归属", "更在意系列完整度"],
        "pay_items": ["IP周边/谷子", "棉花娃娃及配件", "社群活动/展会"],
        "price": "100–300元/月（随追更/补款波动）",
    },
    "佛系水豚": {
        "title": "悦己治愈型用户",
        "traits": ["为自己解压治愈", "重视手感/颜值/质感", "不追热点，偏少而精"],
        "pay_items": ["治愈桌搭摆件", "毛绒/解压小物", "高口碑手办/摆件"],
        "price": "50–300元/月（更看重品质稳定）",
    },
    "理智过客": {
        "title": "理性低涉入型用户",
        "traits": ["不常买，更多试水/送礼", "价格敏感，讨厌溢价", "优先可退换与口碑"],
        "pay_items": ["基础款试水", "礼赠向联名小件", "文具/生活小物"],
        "price": "≤100元/月（节日送礼时上升）",
    },
}

TEAM_INTRO = """
我们是一支关注“童趣消费”与年轻人情绪价值的研究团队。  
我们希望通过轻量测评 + 同好社群 + 分享传播，帮助用户更快找到属于自己的消费人格、
理解自己的购买动机，并在同圈层中获得归属感与表达空间。
"""

ABOUT_US = """
**我们是谁**  
我们是一个围绕“童趣消费人格”展开研究与产品设计的团队，希望把问卷研究、用户画像、圈层社交和传播玩法结合起来。

**为什么做这个**  
很多年轻人的消费并不只是“买东西”，而是在购买治愈感、社交感、收藏感和表达感。  
我们想做一个更有趣、更可分享、也更有研究价值的小应用，把这些差异用可视化和社群互动的方式呈现出来。

**我们希望带来的价值**  
1. 帮助用户更快认识自己的童趣消费偏好  
2. 让同类型用户能在同好广场里互相交流  
3. 用邀请、海报、榜单等方式增强传播和留存
"""

# =========================================================
# 工具函数
# =========================================================
def infer_type(avg: dict) -> str:
    order = sorted(avg.items(), key=lambda x: x[1], reverse=True)
    top = order[0][0]
    second = order[1][0]
    if top == "C":
        return "社交孔雀"
    if top == "A" and second == "C":
        return "圈层海王"
    if top in ["B", "D"]:
        return "佛系水豚"
    return "理智过客"

def blue_color_func(*args, **kwargs):
    h = random.randint(200, 225)
    s = random.randint(55, 90)
    l = random.randint(35, 75)
    return f"hsl({h}, {s}%, {l}%)"

def make_freq_subset(df_words: pd.DataFrame, prefer_words: list[str], top_k=130) -> dict:
    base = dict(zip(df_words["词"].astype(str), df_words["频次"].astype(int)))
    out = {w: base[w] for w in prefer_words if w in base}
    for w, c in base.items():
        if w not in out:
            out[w] = c
        if len(out) >= top_k:
            break
    return out

def render_wordcloud(freq: dict, title: str):
    wc = WordCloud(
        font_path=FONT_PATH if Path(FONT_PATH).exists() else None,
        background_color="white",
        width=1000,
        height=520,
        max_words=150,
        collocations=False,
        color_func=blue_color_func,
    ).generate_from_frequencies(freq)

    st.markdown(f"**{title}**")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)

def adjust_words_by_context(trigger_words, risk_words, budget, premium_ok, scene):
    if budget <= 100:
        risk_words = risk_words + ["价格", "不值", "溢价"]
    if premium_ok == "不接受溢价":
        risk_words = risk_words + ["黄牛", "炒价", "溢价"]
    if scene == "送礼":
        risk_words = risk_words + ["包装", "破损", "退货", "售后", "客服"]
        trigger_words = trigger_words + ["特别", "喜欢", "可爱"]

    def uniq(lst):
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return uniq(trigger_words), uniq(risk_words)

def cutout_checkerboard_to_png(in_path: Path, out_path: Path, tol: int = 22) -> bool:
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        return False

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    mask = np.zeros((h + 2, w + 2), np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    filled = np.zeros((h, w), np.uint8)

    for sx, sy in seeds:
        mask[:] = 0
        ff = blur.copy()
        cv2.floodFill(ff, mask, (sx, sy), 0, loDiff=tol, upDiff=tol)
        region = (mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
        filled = cv2.bitwise_or(filled, region)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.bitwise_not(filled)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = fg

    if fg.mean() < 15:
        return False

    cv2.imwrite(str(out_path), rgba)
    return True

def get_display_image(type_name: str, original_path: Path) -> Path:
    out_png = CUT_DIR / f"{type_name}.png"
    if out_png.exists():
        return out_png
    ok = cutout_checkerboard_to_png(original_path, out_png, tol=22)
    return out_png if ok else original_path

def pil_font(size):
    if Path(FONT_PATH).exists():
        try:
            return ImageFont.truetype(FONT_PATH, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def make_share_poster(user_type: str, one_liner: str) -> bytes:
    W, H = 1080, 1920
    canvas = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    # 背景区块
    draw.rounded_rectangle((50, 50, W-50, H-50), radius=36, fill=(255, 255, 255), outline=(226, 232, 240), width=3)
    draw.rounded_rectangle((80, 80, W-80, 370), radius=30, fill=(239, 246, 255))
    draw.text((120, 130), "测测你的童趣消费人格", font=pil_font(56), fill=(37, 99, 235))
    draw.text((120, 230), f"你的圈层：{user_type}", font=pil_font(76), fill=(15, 23, 42))
    draw.text((120, 330), one_liner, font=pil_font(36), fill=(71, 85, 105))

    # 角色图
    img_path = TYPE_IMG.get(user_type)
    if img_path and img_path.exists():
        show_path = get_display_image(user_type, img_path)
        try:
            role_img = Image.open(show_path).convert("RGBA")
            role_img.thumbnail((700, 700))
            paste_x = (W - role_img.width) // 2
            canvas.paste(role_img, (paste_x, 470), role_img)
        except Exception:
            pass

    # 底部文案
    draw.text((120, 1260), "一句话画像", font=pil_font(40), fill=(236, 72, 153))
    draw.text((120, 1330), TYPE_DESC[user_type], font=pil_font(34), fill=(15, 23, 42))

    draw.text((120, 1490), "扫码 / 截图，邀请好友也来测一测", font=pil_font(34), fill=(100, 116, 139))

    # 小程序码 / 占位
    if MINIPROGRAM_QR.exists():
        qr = Image.open(MINIPROGRAM_QR).convert("RGBA")
        qr.thumbnail((240, 240))
        canvas.paste(qr, (W - 340, 1500), qr)
    else:
        draw.rounded_rectangle((W - 360, 1500, W - 120, 1740), radius=20, fill=(241, 245, 249), outline=(203, 213, 225))
        draw.text((W - 330, 1600), "小程序码\n占位图", font=pil_font(30), fill=(100, 116, 139))

    draw.rounded_rectangle((120, 1600, 540, 1730), radius=26, fill=(249, 168, 212))
    draw.text((160, 1638), "找到你的同好圈层", font=pil_font(36), fill=(131, 24, 67))

    bio = io.BytesIO()
    canvas.save(bio, format="PNG")
    bio.seek(0)
    return bio.getvalue()

def render_persona_card(user_type: str, avatar_img: Image.Image, user_info: dict):
    prof = PERSONA_PROFILE[user_type]

    gender = user_info.get("gender", "—")
    age = user_info.get("age", "—")
    edu = user_info.get("edu", "—")
    job = user_info.get("job", "—")
    city = user_info.get("city", "—")
    income = user_info.get("income", "—")
    purpose = user_info.get("purpose", "—")

    title = prof.get("title", "消费者画像")
    subtitle = "基于测评结果生成的个性化画像卡"

    st.markdown('<div class="profile-wrap">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="profile-head">
          <div>
            <div class="profile-title">{title}</div>
            <div class="profile-sub">{subtitle}</div>
          </div>
          <div class="badge">✨ {user_type}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([0.18, 0.64, 0.18])
    with c2:
        st.image(avatar_img, use_container_width=True)

    colL, colR = st.columns([1.2, 1.0], gap="large")
    with colL:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kv">
              <div class="k">性别</div><div class="v">{gender}</div>
              <div class="k">年龄段</div><div class="v">{age}</div>
              <div class="k">学历</div><div class="v">{edu}</div>
              <div class="k">职业</div><div class="v">{job}</div>
              <div class="k">地区</div><div class="v">{city}</div>
              <div class="k">月可支配</div><div class="v">{income}</div>
              <div class="k">购买目的</div><div class="v">{purpose}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with colR:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        chips = "".join([f"<span class='chip'>{t}</span>" for t in prof.get("traits", [])])
        st.markdown(f"<div class='chips'>{chips}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="mini-card">', unsafe_allow_html=True)
    li = "".join([f"<li>{x}</li>" for x in prof.get("pay_items", [])])
    st.markdown(f"<ul class='list'>{li}</ul>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='kv' style='margin-top:10px;'><div class='k'>典型预算</div><div class='v'>{prof.get('price','—')}</div></div>",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 数据库
# =========================================================
MYSQL_HOST = st.secrets["mysql"]["host"]
MYSQL_PORT = int(st.secrets["mysql"]["port"])
MYSQL_USER = st.secrets["mysql"]["user"]
MYSQL_PASS = st.secrets["mysql"]["password"]
MYSQL_DB = st.secrets["mysql"]["database"]

APP_URL = ""
if "app" in st.secrets and "url" in st.secrets["app"]:
    APP_URL = st.secrets["app"]["url"].rstrip("/")

def get_conn():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )

def init_db():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(50) PRIMARY KEY,
            email VARCHAR(120) UNIQUE,
            nickname VARCHAR(50),
            user_type VARCHAR(50),
            points INT DEFAULT 0,
            invite_code VARCHAR(50) UNIQUE,
            invited_by VARCHAR(50),
            invite_count INT DEFAULT 0,
            joined_at DATETIME
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            post_id INT AUTO_INCREMENT PRIMARY KEY,
            circle_type VARCHAR(50),
            user_id VARCHAR(50),
            content TEXT,
            image LONGBLOB,
            is_featured TINYINT DEFAULT 0,
            created_at DATETIME
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            comment_id INT AUTO_INCREMENT PRIMARY KEY,
            post_id INT,
            user_id VARCHAR(50),
            content TEXT,
            created_at DATETIME,
            FOREIGN KEY (post_id) REFERENCES posts(post_id) ON DELETE CASCADE
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS post_likes (
            like_id INT AUTO_INCREMENT PRIMARY KEY,
            post_id INT,
            user_id VARCHAR(50),
            created_at DATETIME,
            UNIQUE KEY uq_post_user (post_id, user_id)
        )
        """)
    conn.close()

init_db()

# =========================================================
# 读取静态数据
# =========================================================
@st.cache_data
def load_pos_words(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns={"Word": "词", "Count": "频次"})
    return df[["词", "频次"]].sort_values("频次", ascending=False).reset_index(drop=True)

@st.cache_data
def load_neg_words(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df[["词", "频次"]].sort_values("频次", ascending=False).reset_index(drop=True)

pos_df = load_pos_words(POS_XLSX) if POS_XLSX.exists() else pd.DataFrame(columns=["词", "频次"])
neg_df = load_neg_words(NEG_XLSX) if NEG_XLSX.exists() else pd.DataFrame(columns=["词", "频次"])

# =========================================================
# 会话初始化
# =========================================================
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "user_type" not in st.session_state:
    st.session_state["user_type"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "email" not in st.session_state:
    st.session_state["email"] = None
if "invite_pending" not in st.session_state:
    st.session_state["invite_pending"] = None

query_invite = st.query_params.get("invite")
if query_invite and not st.session_state.get("invite_pending"):
    st.session_state["invite_pending"] = str(query_invite)

# =========================================================
# 用户系统
# =========================================================
def fetch_user_by_email(email: str):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        row = cur.fetchone()
    conn.close()
    return row

def fetch_user_by_id(user_id: str):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
    conn.close()
    return row

def fetch_user_by_invite_code(invite_code: str):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE invite_code=%s", (invite_code,))
        row = cur.fetchone()
    conn.close()
    return row

def login_or_register(email: str, nickname: str = ""):
    email = email.strip().lower()
    if not email:
        return None, "请输入邮箱"

    old = fetch_user_by_email(email)
    if old:
        st.session_state["user_id"] = old["user_id"]
        st.session_state["email"] = old["email"]
        st.session_state["user_type"] = old["user_type"]
        return old, "欢迎回来，已自动登录"

    user_id = uuid.uuid4().hex[:16]
    invite_code = uuid.uuid4().hex[:8].upper()

    inviter = None
    pending_code = st.session_state.get("invite_pending")
    if pending_code:
        inviter = fetch_user_by_invite_code(pending_code)

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO users (user_id, email, nickname, user_type, points, invite_code, invited_by, invite_count, joined_at)
            VALUES (%s, %s, %s, %s, 0, %s, NULL, 0, %s)
        """, (user_id, email, nickname.strip(), None, invite_code, datetime.utcnow()))

        if inviter and inviter["user_id"] != user_id:
            cur.execute("""
                UPDATE users
                SET invited_by=%s, points=points+1
                WHERE user_id=%s AND invited_by IS NULL
            """, (inviter["user_id"], user_id))

            cur.execute("""
                UPDATE users
                SET points=points+1, invite_count=invite_count+1
                WHERE user_id=%s
            """, (inviter["user_id"],))

    conn.close()

    new_user = fetch_user_by_email(email)
    st.session_state["user_id"] = new_user["user_id"]
    st.session_state["email"] = new_user["email"]
    st.session_state["user_type"] = new_user["user_type"]
    return new_user, "注册成功"

def update_user_type(user_id: str, user_type: str):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE users
            SET user_type=%s
            WHERE user_id=%s
        """, (user_type, user_id))
    conn.close()
    st.session_state["user_type"] = user_type

def current_user():
    uid = st.session_state.get("user_id")
    if not uid:
        return None
    return fetch_user_by_id(uid)

# =========================================================
# 广场数据函数
# =========================================================
def create_post(circle_type, user_id, content, image_bytes):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO posts (circle_type, user_id, content, image, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (circle_type, user_id, content, image_bytes, datetime.utcnow()))
    conn.close()

def get_posts(circle_type):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.*,
                   COALESCE(u.nickname, u.email, p.user_id) AS author_name,
                   (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.post_id) AS comment_count,
                   (SELECT COUNT(*) FROM post_likes l WHERE l.post_id = p.post_id) AS like_count
            FROM posts p
            LEFT JOIN users u ON p.user_id = u.user_id
            WHERE p.circle_type=%s
            ORDER BY p.is_featured DESC, p.created_at DESC
        """, (circle_type,))
        rows = cur.fetchall()
    conn.close()
    return rows

def add_comment(post_id, user_id, content):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO comments (post_id, user_id, content, created_at)
            VALUES (%s, %s, %s, %s)
        """, (post_id, user_id, content, datetime.utcnow()))
    conn.close()

def get_comments(post_id):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.*, COALESCE(u.nickname, u.email, c.user_id) AS author_name
            FROM comments c
            LEFT JOIN users u ON c.user_id = u.user_id
            WHERE c.post_id=%s
            ORDER BY c.created_at ASC
        """, (post_id,))
        rows = cur.fetchall()
    conn.close()
    return rows

def like_post(post_id, user_id):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT IGNORE INTO post_likes (post_id, user_id, created_at)
            VALUES (%s, %s, %s)
        """, (post_id, user_id, datetime.utcnow()))
    conn.close()

def has_liked(post_id, user_id):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM post_likes WHERE post_id=%s AND user_id=%s", (post_id, user_id))
        row = cur.fetchone()
    conn.close()
    return row is not None

def get_month_best_post(circle_type):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.*,
                   COALESCE(u.nickname, u.email, p.user_id) AS author_name,
                   COUNT(l.like_id) AS like_count
            FROM posts p
            LEFT JOIN post_likes l ON p.post_id = l.post_id
            LEFT JOIN users u ON p.user_id = u.user_id
            WHERE p.circle_type=%s
              AND DATE_FORMAT(p.created_at, '%%Y-%%m') = DATE_FORMAT(CURRENT_DATE(), '%%Y-%%m')
            GROUP BY p.post_id
            ORDER BY like_count DESC, p.created_at DESC
            LIMIT 1
        """, (circle_type,))
        row = cur.fetchone()
    conn.close()
    return row

# =========================================================
# 导航
# =========================================================
def go(page):
    st.session_state["page"] = page
    st.rerun()

with st.container():
    st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
    with c1:
        st.markdown("### 童趣消费人格")
    with c2:
        if st.button("首页", use_container_width=True):
            go("home")
    with c3:
        if st.button("测一测", use_container_width=True):
            go("quiz")
    with c4:
        if st.button("同好广场", use_container_width=True):
            go("plaza")
    with c5:
        if st.button("关于我们", use_container_width=True):
            go("about")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 公共登录卡
# =========================================================
def render_login_card(prefix="default"):
    user = current_user()
    if user:
        st.success(f"当前已绑定：{user['email']}")
        return user

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 先绑定邮箱")
    st.caption("用于唯一识别用户、记录积分、邀请关系和社区互动。")

    email = st.text_input("邮箱", key=f"{prefix}_bind_email")
    nickname = st.text_input("昵称（可选）", key=f"{prefix}_bind_nickname")

    if st.button("绑定 / 登录", key=f"{prefix}_bind_email_btn", use_container_width=True):
        user, msg = login_or_register(email, nickname)
        if user:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    st.markdown('</div>', unsafe_allow_html=True)
    return None
# =========================================================
# 首页
# =========================================================
def render_home():
    st.markdown(
        """
        <div class="hero-wrap">
          <div class="hero-title">测测你的童趣消费人格</div>
          <div class="hero-sub">1分钟查看你的童趣消费人格，并获得圈层画像、同好推荐、分享海报与专属邀请链接。</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 队伍简介")
    st.write(TEAM_INTRO)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## 四大圈层快速预览")
    cols = st.columns(4)
    for i, t in enumerate(TYPE_DESC.keys()):
        with cols[i]:
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.markdown(f"### {t}")
            st.caption(TYPE_DESC[t])
            st.markdown("代表词：" + "、".join(TYPE_PREVIEW_WORDS[t]))
            if TYPE_IMG[t].exists():
                st.image(Image.open(get_display_image(t, TYPE_IMG[t])), use_container_width=True)
            if st.button(f"查看 {t}", key=f"preview_{t}", use_container_width=True):
                st.session_state["preview_type"] = t
            st.markdown('</div>', unsafe_allow_html=True)

    ptype = st.session_state.get("preview_type")
    if ptype:
        st.markdown("---")
        st.markdown(f"### {ptype}画像预览")
        st.write(TYPE_DESC[ptype])
        st.markdown("代表词：" + "、".join(TYPE_PREVIEW_WORDS[ptype]))

    st.markdown("---")
    a, b, c = st.columns(3)
    with a:
        if st.button("测一测", key="home_to_quiz", use_container_width=True):
            go("quiz")
    with b:
        if st.button("同好广场", key="home_to_plaza", use_container_width=True):
            go("plaza")
    with c:
        if st.button("关于我们", key="home_to_about", use_container_width=True):
            go("about")

# =========================================================
# 测评页
# =========================================================
def render_quiz():
    st.markdown("## 开始测评")
    st.caption("1 表示非常不同意，5 表示非常同意。")

    ans = {"A": [], "B": [], "C": [], "D": []}
    for i, (q, grp) in enumerate(QUESTIONS, start=1):
        v = st.slider(f"{i}. {q}", 1, 5, 3, key=f"quiz_{i}")
        ans[grp].append(v)

    if st.button("开始测评", use_container_width=True):
        with st.spinner("正在分析你的童趣人格..."):
            time.sleep(2)
            avg = {k: float(np.mean(v)) for k, v in ans.items()}
            user_type = infer_type(avg)
            st.session_state["user_type"] = user_type
            user = current_user()
            if user:
                update_user_type(user["user_id"], user_type)
            st.session_state["page"] = "result"
            st.rerun()

# =========================================================
# 结果页
# =========================================================
def render_result():
    user_type = st.session_state.get("user_type")
    if not user_type:
        st.info("你还没有完成测评，先去测一测吧。")
        if st.button("去测评", use_container_width=True):
            go("quiz")
        return

    cfg = RECO[user_type]
    user = render_login_card(prefix="result")

    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 你的童趣消费人格")
        st.markdown(f"<div><span class='pill'>{user_type}</span></div>", unsafe_allow_html=True)
        img_path = TYPE_IMG.get(user_type)
        if img_path and img_path.exists():
            st.image(Image.open(get_display_image(user_type, img_path)), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 一句话画像")
        st.write(TYPE_DESC[user_type])
        st.markdown("### 你的同好方向")
        st.markdown(f"<span class='pill2'>{cfg['tribe']}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    b1, b2 = st.columns(2)
    with b1:
        open_profile = st.button("查看我的画像卡", use_container_width=True)
    with b2:
        open_reco = st.button("可能这里有你的同好", use_container_width=True)

    if open_profile:
        @st.dialog("消费者画像卡")
        def show_profile():
            st.write("补充信息可让画像更贴合你本人（不填也能生成）。")
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("性别", ["—", "男", "女", "其他/不便透露"])
                age = st.selectbox("年龄段", ["—", "18-24", "25-30", "31-40", "41+"])
                income = st.selectbox("月可支配收入", ["—", "≤5000", "5001-9999", "10000-19999", "20000+"])
            with c2:
                edu = st.selectbox("学历", ["—", "大专及以下", "本科", "硕士及以上"])
                job = st.selectbox("职业", ["—", "学生", "企业职员", "自由职业", "事业单位/公务员", "其他"])
                city = st.selectbox("地区", ["—", "一线/新一线", "二线", "三线及以下"])
            purpose = st.selectbox("购买目的", ["—", "自我治愈/解压", "社交分享", "圈层收集", "送礼"])

            user_info = {
                "gender": gender, "age": age, "edu": edu, "job": job,
                "city": city, "income": income, "purpose": purpose
            }

            img_path0 = TYPE_IMG.get(user_type)
            if img_path0 and img_path0.exists():
                avatar = Image.open(get_display_image(user_type, img_path0))
                render_persona_card(user_type, avatar, user_info)

        show_profile()

    if open_reco:
        @st.dialog("同好推荐与选品分析")
        def show_reco():
            st.markdown(f"### 你的类型：{user_type}")
            img_path2 = TYPE_IMG.get(user_type)
            if img_path2 and img_path2.exists():
                st.image(Image.open(get_display_image(user_type, img_path2)), width=240)

            st.markdown(f"<span class='pill2'>{cfg['tribe']}</span>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                budget = st.slider("预算（元）", 50, 1000, 200, step=50)
            with c2:
                premium_ok = st.selectbox("对溢价的态度", ["不接受溢价", "适度接受", "可接受较高溢价"])
            with c3:
                scene = st.selectbox("主要场景", ["自我治愈", "社交分享", "圈层收集", "送礼"])

            st.markdown("### 推荐品牌 / 产品方向")
            for b in cfg["brands_primary"]:
                st.markdown(f"- {b}")

            st.markdown("### 建议购买方式")
            for c in cfg["channels"]:
                st.markdown(f"- {c}")

            st.markdown("### 选择一个方向")
            items = cfg["items"]
            options = [f"{i+1}. {it['title']}" for i, it in enumerate(items)]
            pick = st.selectbox("方向", options)
            it = items[options.index(pick)]

            st.markdown(
                f"<div class='card'><b>{it['title']}</b><br><span class='muted'>{it['why']}</span>"
                f"<br><br>关键词：{'、'.join(it['types'])}</div>",
                unsafe_allow_html=True
            )

            trigger_words, risk_words = adjust_words_by_context(
                it["trigger_words"], it["risk_words"], budget, premium_ok, "送礼" if scene == "送礼" else scene
            )

            st.divider()
            L, R = st.columns(2)
            with L:
                pos_freq = make_freq_subset(pos_df, trigger_words, top_k=130)
                render_wordcloud(pos_freq, "你可能会被这些点打动")
            with R:
                neg_freq = make_freq_subset(neg_df, risk_words, top_k=130)
                render_wordcloud(neg_freq, "你更可能踩雷的点（建议重点避坑）")

        show_reco()

    st.divider()
    x1, x2 = st.columns(2)

    with x1:
        poster_bytes = make_share_poster(user_type, TYPE_DESC[user_type])
        st.download_button(
            "生成分享海报",
            data=poster_bytes,
            file_name=f"{user_type}_分享海报.png",
            mime="image/png",
            use_container_width=True
        )

    with x2:
        user = current_user()
        if user and APP_URL:
            invite_link = f"{APP_URL}?invite={user['invite_code']}"
            st.code(invite_link)
            st.caption("复制上方专属邀请链接，好友完成绑定后，双方各得 1 点同好值。")
        else:
            st.info("先绑定邮箱，并在 secrets 中配置 app.url，即可生成邀请链接。")

# =========================================================
# 个人中心
# =========================================================
def render_profile_center():
    st.markdown("## 个人中心")
    user = render_login_card(prefix="profile")
    if not user:
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 我的信息")
    st.markdown(
        f"""
        <div class="kv">
          <div class="k">邮箱</div><div class="v">{user['email']}</div>
          <div class="k">当前圈层</div><div class="v">{user['user_type'] or '未完成测评'}</div>
          <div class="k">同好值</div><div class="v">{user['points']}</div>
          <div class="k">邀请人数</div><div class="v">{user['invite_count']}</div>
          <div class="k">邀请码</div><div class="v">{user['invite_code']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if APP_URL:
        invite_link = f"{APP_URL}?invite={user['invite_code']}"
        st.markdown("### 我的专属邀请链接")
        st.code(invite_link)

# =========================================================
# 同好广场
# =========================================================
def render_plaza():
    st.markdown("## 同好广场")
    user = render_login_card(prefix="plaza")
    if not user:
        return

    if not user["user_type"]:
        st.warning("先完成测评，系统会自动将你加入对应圈层。")
        if st.button("去完成测评", use_container_width=True):
            go("quiz")
        return

    st.caption("完成测评后将自动入圈，并默认进入你的专属讨论区。")

    CIRCLE_TYPES = ["社交孔雀", "圈层海王", "佛系水豚", "理智过客"]
    default_idx = CIRCLE_TYPES.index(user["user_type"]) if user["user_type"] in CIRCLE_TYPES else 0
    selected_circle = st.selectbox("选择圈层分区", CIRCLE_TYPES, index=default_idx)

    best = get_month_best_post(selected_circle)
    if best:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 本月最佳晒单")
        st.caption("按本月点赞数自动展示，可用于后续加精置顶 / 发放虚拟勋章。")
        st.markdown(f"**{best['author_name']}**")
        st.write(best["content"])
        st.markdown(f"👍 {best['like_count']} 赞")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"### {selected_circle}区 · 发帖")
    with st.form("post_form", clear_on_submit=True):
        post_text = st.text_area("写下你的分享 / 评论 / 晒单")
        post_image = st.file_uploader("上传图片（可选）", type=["png", "jpg", "jpeg"])
        submit_post = st.form_submit_button("发帖")
        if submit_post:
            if not post_text.strip() and not post_image:
                st.warning("至少输入文字或上传图片中的一种。")
            else:
                image_bytes = post_image.read() if post_image else None
                create_post(selected_circle, user["user_id"], post_text.strip(), image_bytes)
                st.success("发帖成功！")
                st.rerun()

    st.markdown(f"### {selected_circle}区 · 帖子列表")
    posts = get_posts(selected_circle)
    if not posts:
        st.info("这个圈层还没有帖子，发第一条试试吧。")
        return

    for post in posts:
        st.markdown("---")
        badge = " 🏅加精" if post.get("is_featured") else ""
        st.markdown(f"**{post['author_name']}** 发表于 {post['created_at'].strftime('%Y-%m-%d %H:%M:%S')}{badge}")
        if post.get("content"):
            st.write(post["content"])
        if post.get("image"):
            st.image(Image.open(io.BytesIO(post["image"])), use_container_width=True)

        like_disabled = has_liked(post["post_id"], user["user_id"])
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button(
                "❤️ 点赞",
                key=f"like_btn_{post['post_id']}",
                disabled=like_disabled,
                use_container_width=True
            ):
                like_post(post["post_id"], user["user_id"])
                st.rerun()
        with c2:
            st.markdown(f"**{post['like_count']} 人点赞 · {post['comment_count']} 条评论**")

        st.markdown("**评论：**")
        comments = get_comments(post["post_id"])
        if comments:
            for cm in comments:
                st.markdown(f"- **{cm['author_name']}**：{cm['content']}")
        else:
            st.caption("还没有评论。")

        with st.form(f"comment_form_{post['post_id']}", clear_on_submit=True):
            comment_text = st.text_input("写评论")
            submit_comment = st.form_submit_button("提交评论")
            if submit_comment and comment_text.strip():
                add_comment(post["post_id"], user["user_id"], comment_text.strip())
                st.success("评论成功！")
                st.rerun()

# =========================================================
# 关于我们
# =========================================================
def render_about():
    st.markdown("## 关于我们")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(ABOUT_US)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 页面路由
# =========================================================
page = st.session_state["page"]

if page == "home":
    render_home()
elif page == "quiz":
    render_quiz()
elif page == "result":
    render_result()
elif page == "plaza":
    render_plaza()
elif page == "about":
    render_about()
elif page == "profile":
    render_profile_center()
else:
    render_home()

# st.divider()
# render_profile_center()
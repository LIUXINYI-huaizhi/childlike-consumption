import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# -----------------------------
# 基础设置
# -----------------------------
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

POS_XLSX = DATA_DIR / "pos_top_words.xlsx"         # Word / Count
NEG_XLSX = DATA_DIR / "neg_top_words_new.xlsx"     # 词 / 频次

# 字体：优先项目内字体（部署更稳），否则用Windows系统字体
FONT_PATH = str(FONT_DIR / "msyh.ttc")
if not Path(FONT_PATH).exists():
    FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"

# -----------------------------
# 页面美化 CSS（解决标题遮挡 + 居中 + 更好看字体）
# -----------------------------
st.markdown(
    """
    <style>
    /* 给顶部留空间，避免标题被挡 */
    .block-container {
        padding-top: 3.2rem !important;
        padding-bottom: 2.2rem;
        max-width: 1100px;
    }

    /* 弱化/隐藏 Streamlit 自带头部（避免压住内容） */
    header[data-testid="stHeader"] { 
        background: rgba(255,255,255,0);
        height: 0px;
    }
    div[data-testid="stToolbar"] { visibility: hidden; height: 0px; }

    /* 全局字体更舒服（不依赖外网字体） */
    html, body, [class*="css"]  {
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI",
                     "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei",
                     Arial, sans-serif;
    }

    /* hero区居中 */
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

    /* 卡片容器 */
    .card {
        border: 1px solid rgba(148,163,184,.35);
        border-radius: 18px;
        padding: 16px 16px;
        background: rgba(255,255,255,.82);
        box-shadow: 0 8px 24px rgba(2,6,23,.08);
    }
    .muted { color: #64748b; }

    /* 胶囊标签 */
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

    /* 按钮更圆润 */
    div.stButton>button {
        border-radius: 14px;
        padding: 0.7rem 1.0rem;
        font-weight: 800;
        border: 1px solid rgba(148,163,184,.55);
        box-shadow: 0 10px 26px rgba(2,6,23,.10);
    }
    div.stButton>button:hover { transform: translateY(-1px); }

    /* slider 间距更紧凑 */
    .stSlider { padding-top: .15rem; padding-bottom: .15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* ============ 画像卡美化 ============ */
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
    .badge i{ opacity:.9; font-style:normal; }

    .mini-card{
        background: rgba(255,255,255,.85);
        border: 1px solid rgba(148,163,184,.28);
        border-radius: 18px;
        padding: 14px 14px;
        box-shadow: 0 10px 22px rgba(2,6,23,.06);
        margin-bottom: 12px;
    }
    .sec-title{
        font-size: 1.05rem;
        font-weight: 900;
        margin: 0 0 10px 0;
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

    .list{
        margin: 0;
        padding-left: 1.0rem;
    }
    .list li{
        margin: .35rem 0;
        line-height: 1.55;
        font-size: .98rem;
        color:#0f172a;
    }

    .right-card{
        background: rgba(255,255,255,.85);
        border: 1px solid rgba(148,163,184,.28);
        border-radius: 18px;
        padding: 14px 14px;
        box-shadow: 0 10px 22px rgba(2,6,23,.06);
        height: 100%;
    }
    .img-title{
        font-size: 1.02rem;
        font-weight: 900;
        margin: 0 0 10px 0;
    }

    /* 控制图片更好看：居中 + 柔和阴影 */
    .stImage img{
        border-radius: 16px;
        box-shadow: 0 16px 40px rgba(2,6,23,.12);
    }
    </style>
    """,
    unsafe_allow_html=True
)
# -----------------------------
# 抠像：把棋盘格背景变透明 PNG
# -----------------------------
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
        lo = tol
        up = tol
        cv2.floodFill(ff, mask, (sx, sy), 0, loDiff=lo, upDiff=up)
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


# -----------------------------
# 读词频表
# -----------------------------
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


# -----------------------------
# 测评题库
# -----------------------------
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

TYPE_IMG = {
    "社交孔雀": IMG_DIR / "social_peacock.jpg",
    "圈层海王": IMG_DIR / "circle_sea.jpg",
    "佛系水豚": IMG_DIR / "chill_cap.jpg",
    "理智过客": IMG_DIR / "rational_guest.jpg",
}

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


# -----------------------------
# 推荐库
# -----------------------------
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

# -----------------------------
# 消费者画像模板（用于画像卡）
# -----------------------------
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

    # ===== 标题行 =====
    st.markdown(
        f"""
        <div class="profile-head">
          <div>
            <div class="profile-title">{title}</div>
            <div class="profile-sub">{subtitle}</div>
          </div>
          <div class="badge"><i>✨</i>{user_type}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===== 形象示意：单独一行（标题下方），放大居中 =====
    st.markdown('<div class="right-card" style="padding:18px; margin-bottom: 12px;">', unsafe_allow_html=True)
    st.markdown('<div class="img-title" style="margin-bottom:12px;">形象示意</div>', unsafe_allow_html=True)

    # 居中 + 放大：中间列更宽
    c1, c2, c3 = st.columns([0.18, 0.64, 0.18])
    with c2:
        st.image(avatar_img, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ===== 基本信息 & 核心特征：左右两列 =====
    colL, colR = st.columns([1.2, 1.0], gap="large")

    with colL:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">基本信息</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="sec-title">核心特征</div>', unsafe_allow_html=True)
        chips = "".join([f"<span class='chip'>{t}</span>" for t in prof.get("traits", [])])
        st.markdown(f"<div class='chips'>{chips}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ===== 常见购买项目与价格带：单独一行 =====
    st.markdown('<div class="mini-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">常见购买项目与价格带</div>', unsafe_allow_html=True)

    items = prof.get("pay_items", [])
    if items:
        li = "".join([f"<li>{x}</li>" for x in items])
        st.markdown(f"<ul class='list'>{li}</ul>", unsafe_allow_html=True)

    budget = prof.get("price", "—")
    st.markdown(
        f"<div class='kv' style='margin-top:10px;'><div class='k'>典型预算</div><div class='v'>{budget}</div></div>",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
# -----------------------------
# 词云工具
# -----------------------------
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
        font_path=FONT_PATH,
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


# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <div class="hero-wrap">
      <div class="hero-title">测测你的童趣消费人格</div>
      <div class="hero-sub">1分钟查看你的童趣消费人格，并获得同好推荐、选品分析与词云避坑。</div>
    </div>
    """,
    unsafe_allow_html=True
)

if pos_df.empty or neg_df.empty:
    st.error("没有读到词频表。请检查 data/ 下是否存在 pos_top_words.xlsx 与 neg_top_words_new.xlsx。")
    st.stop()

# -----------------------------
# 测评区
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("## 先回答几道题")
st.markdown('<div class="muted">1 表示非常不同意，5 表示非常同意。</div>', unsafe_allow_html=True)

ans = {"A": [], "B": [], "C": [], "D": []}
for i, (q, grp) in enumerate(QUESTIONS, start=1):
    v = st.slider(f"{i}. {q}", 1, 5, 3)
    ans[grp].append(v)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 按钮：查看我的人格
# -----------------------------
clicked = st.button("查看我的人格", use_container_width=True)

if clicked:
    avg = {k: float(np.mean(v)) for k, v in ans.items()}
    user_type = infer_type(avg)
    st.session_state["user_type"] = user_type

user_type = st.session_state.get("user_type", None)

# -----------------------------
# 结果卡片
# -----------------------------
if user_type:
    cfg = RECO[user_type]

    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 你的童趣消费人格")
        st.markdown(f"<div><span class='pill'>{user_type}</span></div>", unsafe_allow_html=True)

        img_path = TYPE_IMG.get(user_type)
        if img_path and img_path.exists():
            show_path = get_display_image(user_type, img_path)
            st.image(Image.open(show_path), use_container_width=True)
        else:
            st.info("把该人格的卡通图放入 images/ 目录即可显示。")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 一句话画像")
        st.write(TYPE_DESC[user_type])
        st.markdown("### 下一步")
        st.markdown(f"<span class='pill2'>{cfg['tribe']}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # 两个按钮：画像卡 + 同好推荐
    b1, b2 = st.columns(2)
    with b1:
        open_profile = st.button("查看我的画像卡", use_container_width=True)
    with b2:
        open_reco = st.button("可能这里有你的同好", use_container_width=True)

    # 画像卡弹窗
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
            if not (img_path0 and img_path0.exists()):
                st.warning("未找到该人格的形象图，请检查 images/ 目录。")
                return

            show_path0 = get_display_image(user_type, img_path0)
            avatar = Image.open(show_path0)

            render_persona_card(user_type, avatar, user_info)

        show_profile()

    # 同好弹窗
    if open_reco:
        @st.dialog("同好推荐与选品分析")
        def show_reco():
            st.markdown(f"### 你的类型：{user_type}")

            img_path2 = TYPE_IMG.get(user_type)
            if img_path2 and img_path2.exists():
                show_path2 = get_display_image(user_type, img_path2)
                st.image(Image.open(show_path2), width=240)

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

            st.markdown(f"<div class='card'><b>{it['title']}</b><br><span class='muted'>{it['why']}</span>"
                        f"<br><br>关键词：{'、'.join(it['types'])}</div>", unsafe_allow_html=True)

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

            st.caption("词云由你研究文本数据的高频表达生成，用于解释推荐理由与常见风险。")

        show_reco()
# =============================
# MySQL 配置（同好广场）
# =============================
import pymysql
import io
from datetime import datetime

MYSQL_HOST = st.secrets["mysql"]["host"]
MYSQL_PORT = int(st.secrets["mysql"]["port"])
MYSQL_USER = st.secrets["mysql"]["user"]
MYSQL_PASS = st.secrets["mysql"]["password"]
MYSQL_DB   = st.secrets["mysql"]["database"]

# 创建连接
conn = pymysql.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASS,
    database=MYSQL_DB,
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True
)

# -----------------------------
# 初始化表（第一次运行可执行）
# -----------------------------
with conn.cursor() as cur:
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(50) PRIMARY KEY,
        user_type VARCHAR(50),
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
        created_at DATETIME,
        likes INT DEFAULT 0
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

# -----------------------------
# 同好广场模块
# -----------------------------
if user_type:
    st.markdown("## 同好广场")

    # 自动入圈
    user_id = st.session_state.get("user_id", f"user_{random.randint(1000,9999)}")
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO users (user_id, user_type, joined_at)
            VALUES (%s,%s,%s)
            ON DUPLICATE KEY UPDATE user_type=%s, joined_at=%s
        """, (user_id, user_type, datetime.utcnow(), user_type, datetime.utcnow()))

    # 圈层选择
    CIRCLE_TYPES = ["社交孔雀","圈层海王","佛系水豚","理智过客"]
    selected_circle = st.selectbox("圈层选择", CIRCLE_TYPES, index=CIRCLE_TYPES.index(user_type))

    # 发帖功能
    st.markdown(f"### {selected_circle}区 - 发帖")
    with st.form("post_form", clear_on_submit=True):
        post_text = st.text_area("写下你的分享/评论")
        post_image = st.file_uploader("上传图片 (可选)", type=["png","jpg","jpeg"])
        submit_post = st.form_submit_button("发帖")
        if submit_post:
            image_bytes = post_image.read() if post_image else None
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO posts (circle_type, user_id, content, image, created_at, likes)
                    VALUES (%s,%s,%s,%s,%s,0)
                """, (selected_circle, user_id, post_text, image_bytes, datetime.utcnow()))
            st.success("发帖成功！")

    # 展示帖子列表
    st.markdown(f"### {selected_circle}区 - 帖子列表")
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("SELECT * FROM posts WHERE circle_type=%s ORDER BY created_at DESC", (selected_circle,))
        posts = cur.fetchall()

    # 点赞回调
    def make_like_callback(post_id, state_key):
        def callback():
            with conn.cursor() as cur:
                cur.execute("UPDATE posts SET likes=likes+1 WHERE post_id=%s", (post_id,))
            st.session_state[state_key] = True
        return callback

    for post in posts:
        st.markdown("---")
        st.markdown(f"**{post.get('user_id','匿名')}** 发表于 {post['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(post.get("content",""))
        if post.get("image"):
            image = Image.open(io.BytesIO(post["image"]))
            st.image(image, use_container_width=True)

        # 点赞按钮与数量
        like_state_key = f"liked_{post['post_id']}"   # 状态 key
        button_key = f"like_btn_{post['post_id']}"     # 按钮 key
        if like_state_key not in st.session_state:
            st.session_state[like_state_key] = False

        with st.container():
            c1, c2 = st.columns([1, 5])
            with c1:
                st.button(
                    "❤️ 点赞",
                    key=button_key,
                    on_click=make_like_callback(post['post_id'], like_state_key),
                    disabled=st.session_state[like_state_key]
                )
            with c2:
                with conn.cursor() as cur:
                    # 点赞数
                    cur.execute("SELECT likes FROM posts WHERE post_id=%s", (post['post_id'],))
                    likes_count = cur.fetchone()["likes"]

                    # 评论数
                    cur.execute("SELECT COUNT(*) AS comment_count FROM comments WHERE post_id=%s", (post['post_id'],))
                    comments_count = cur.fetchone()["comment_count"]
                st.markdown(f"**{likes_count} 人点赞 · {comments_count} 条评论**")

        # 评论区
        st.markdown("**评论:**")
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM comments WHERE post_id=%s ORDER BY created_at ASC", (post['post_id'],))
            post_comments = cur.fetchall()
        for comment in post_comments:
            st.markdown(f"- **{comment.get('user_id','匿名')}**: {comment['content']}")

        # 评论表单
        with st.form(f"comment_form_{post['post_id']}", clear_on_submit=True):
            comment_text = st.text_input("写评论")
            submit_comment = st.form_submit_button("提交评论")
            if submit_comment and comment_text:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO comments (post_id, user_id, content, created_at)
                        VALUES (%s,%s,%s,%s)
                    """, (post['post_id'], user_id, comment_text, datetime.utcnow()))
                st.success("评论成功！")
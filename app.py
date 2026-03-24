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
        "brands_primary": ["泡泡玛特（POP MART）", "TOP TOY（名创优品旗下）", "52TOYS", "迪士尼", "Medicom Toy"],
        "channels": ["官方旗舰店/小程序", "线下门店与快闪", "正规电商自营（减少黄牛风险）"],
        "same_circle_buys": [
            {"name": "MEGA SPACE MOLLY 1000%", "brand": "泡泡玛特", "price_range": ">1000元", "tag": "高调展示"},
            {"name": "KAWS 假日限定玩偶", "brand": "KAWS", "price_range": ">1000元", "tag": "身份象征"},
            {"name": "Sonny Angel 系列盲盒", "brand": "Sonny Angel", "price_range": "<100元", "tag": "经典拆盒"},
            {"name": "泡泡玛特 哈利波特系列", "brand": "泡泡玛特", "price_range": "<100元", "tag": "IP热度高"},
        ],
        "pitfalls": [
            "小心“网红爆款”溢价过高，多平台比价再入手。",
            "晒图时注意隐私，避免暴露家庭地址等信息。"
        ],
        "items": [
            {
                "title": "炫耀感强的限量潮玩",
                "types": ["盲盒", "限量", "社交货币"],
                "why": "适合爱分享、爱晒图、享受“拥有稀缺款”成就感的人。",
                "trigger_words": ["抽到", "隐藏款", "限定", "联名", "喜欢", "可爱", "特别"],
                "risk_words": ["概率", "重复", "黄牛", "炒价", "溢价", "不值"],
                "products": [
                    {
                        "name": "LABUBU 一代坐坐派对",
                        "brand": "泡泡玛特",
                        "price_range": "<100元",
                        "emotion_tags": ["限量", "社交货币"],
                        "reason": "盲盒形式，隐藏款稀有，适合晒图炫耀。"
                    },
                    {
                        "name": "BE@RBRICK 100% 限定款",
                        "brand": "Medicom Toy",
                        "price_range": ">1000元",
                        "emotion_tags": ["稀缺", "收藏"],
                        "reason": "积木熊本身就是潮玩圈的身份象征，晒出来倍有面。"
                    },
                    {
                        "name": "迪士尼 玲娜贝儿中秋限定公仔",
                        "brand": "迪士尼",
                        "price_range": "500-1000元",
                        "emotion_tags": ["社交货币", "节日限定"],
                        "reason": "限定款难抢，拥有即证明“手速+财力”。"
                    }
                ]
            },
            {
                "title": "适合内容分享的设计感产品",
                "types": ["设计感", "评测向", "内容素材"],
                "why": "适合喜欢种草、安利、做内容输出的人。",
                "trigger_words": ["设计", "独特", "分享", "内容", "推荐", "种草"],
                "risk_words": ["翻车", "做工", "瑕疵", "广告", "信任"],
                "products": [
                    {
                        "name": "SKULLPANDA 温度系列",
                        "brand": "泡泡玛特",
                        "price_range": "<100元",
                        "emotion_tags": ["审美", "专业"],
                        "reason": "设计感强，适合作为内容素材分享给粉丝。"
                    },
                    {
                        "name": "FARMER BOB 限定款",
                        "brand": "寻找独角兽",
                        "price_range": "500-1000元",
                        "emotion_tags": ["引领", "收藏"],
                        "reason": "BOB 在圈内辨识度高，适合做KOL展示。"
                    },
                    {
                        "name": "52TOYS 猛兽匣系列",
                        "brand": "52TOYS",
                        "price_range": "100-300元",
                        "emotion_tags": ["创新", "可玩性"],
                        "reason": "变形玩法独特，适合做深度评测。"
                    }
                ]
            },
            {
                "title": "适合开箱表演和互动展示",
                "types": ["惊喜", "仪式感", "互动玩法"],
                "why": "适合喜欢拆盒过程、享受氛围感和展示效果的人。",
                "trigger_words": ["拆盒", "惊喜", "仪式感", "互动", "直播", "展示"],
                "risk_words": ["剧透", "重复", "概率", "遮挡", "黄牛"],
                "products": [
                    {
                        "name": "Dimoo 童话系列盲盒",
                        "brand": "泡泡玛特",
                        "price_range": "<100元",
                        "emotion_tags": ["惊喜", "仪式感"],
                        "reason": "拆盒过程充满戏剧性，适合直播。"
                    },
                    {
                        "name": "若来 超级元气工厂 拆箱盲袋",
                        "brand": "若来",
                        "price_range": "100-300元",
                        "emotion_tags": ["表演", "惊喜"],
                        "reason": "盲袋形式，层层拆解，直播效果拉满。"
                    },
                    {
                        "name": "迪士尼 奇奇蒂蒂 摇摇乐摆件",
                        "brand": "迪士尼",
                        "price_range": "100-300元",
                        "emotion_tags": ["互动", "可爱"],
                        "reason": "可以边拆边演示互动玩法。"
                    }
                ]
            },
            {
                "title": "热门联名和跟风不易出错款",
                "types": ["热门", "联名", "大众流行"],
                "why": "适合想快速跟上热点、买到“不会错”的爆款用户。",
                "trigger_words": ["热门", "联名", "爆款", "流行", "这次", "可爱"],
                "risk_words": ["饥饿营销", "溢价", "冲动消费", "排队", "断货"],
                "products": [
                    {
                        "name": "Molly 每日故宫系列",
                        "brand": "泡泡玛特",
                        "price_range": "<100元",
                        "emotion_tags": ["热门", "文化"],
                        "reason": "故宫IP当下热度高，适合跟风。"
                    },
                    {
                        "name": "LINE FRIENDS 布朗熊 毛绒公仔",
                        "brand": "LINE",
                        "price_range": "100-300元",
                        "emotion_tags": ["流行", "大众"],
                        "reason": "国民级IP，跟风不出错。"
                    },
                    {
                        "name": "名创优品 三丽鸥联名盲袋",
                        "brand": "名创优品",
                        "price_range": "<100元",
                        "emotion_tags": ["平价", "热门"],
                        "reason": "价格亲民，紧跟联名热潮。"
                    }
                ]
            }
        ]
    },

    "圈层海王": {
        "tribe": "或许这里有你的同好：谷圈、娃圈、IP收藏党、同好社群",
        "brands_primary": ["泡泡玛特（联名线）", "名创优品联名", "官方IP店/旗舰店", "迪士尼", "万代", "三丽鸥"],
        "channels": ["官方授权渠道（减少盗版）", "同好社群团购需谨慎", "线下IP主题店"],
        "same_circle_buys": [
            {"name": "龙珠 Z 超像Art 系列", "brand": "万代", "price_range": "500-1000元", "tag": "怀旧动漫"},
            {"name": "哆啦A梦 秘密道具 扭蛋", "brand": "万代", "price_range": "<100元", "tag": "童年记忆"},
            {"name": "三丽鸥 双星仙子 周边", "brand": "三丽鸥", "price_range": "<100元", "tag": "同好社交"},
            {"name": "EVA 初号机 手办", "brand": "EVA", "price_range": "500-1000元", "tag": "机甲收藏"},
            {"name": "北京环球影城 小黄人 爆米花桶", "brand": "环球影城", "price_range": "100-300元", "tag": "园区限定"},
        ],
        "pitfalls": [
            "别为了“集齐”而超预算，圈层归属感不等于必须全入。",
            "优先走官方授权渠道，避免盗版和二手高溢价。"
        ],
        "items": [
            {
                "title": "怀旧向 IP 和童年回忆款",
                "types": ["童年", "情怀", "纪念"],
                "why": "适合被童年记忆打动、愿意为老IP持续买单的人。",
                "trigger_words": ["童年", "回忆", "情怀", "纪念", "经典"],
                "risk_words": ["复刻缩水", "溢价", "占空间", "冲动囤货"],
                "products": [
                    {
                        "name": "宝可梦 皮卡丘 20周年复刻公仔",
                        "brand": "万代",
                        "price_range": "300-500元",
                        "emotion_tags": ["童年", "情怀"],
                        "reason": "20周年复刻版，满满的童年回忆。"
                    },
                    {
                        "name": "小浣熊 水浒卡 108将 收藏卡",
                        "brand": "统一",
                        "price_range": "<100元",
                        "emotion_tags": ["怀旧", "集卡"],
                        "reason": "干脆面卡片是80/90后集体记忆。"
                    },
                    {
                        "name": "Hello Kitty 50周年纪念款",
                        "brand": "三丽鸥",
                        "price_range": "100-300元",
                        "emotion_tags": ["情怀", "纪念"],
                        "reason": "周年限定，承载多年陪伴。"
                    }
                ]
            },
            {
                "title": "适合同好社交和交换互动的周边",
                "types": ["社交", "交换", "同好互动"],
                "why": "适合通过收藏认识朋友、参加线下交换和社群互动的人。",
                "trigger_words": ["同好", "交换", "社交", "挂件", "交流"],
                "risk_words": ["攀比", "黄牛", "溢价", "线下安全"],
                "products": [
                    {
                        "name": "迪士尼 达菲与朋友们 挂件",
                        "brand": "迪士尼",
                        "price_range": "100-300元",
                        "emotion_tags": ["同好", "社交"],
                        "reason": "达菲家族是社交利器，线下换娃必备。"
                    },
                    {
                        "name": "泡泡玛特 萌粒 糖罐",
                        "brand": "泡泡玛特",
                        "price_range": "<100元",
                        "emotion_tags": ["交换", "互动"],
                        "reason": "萌粒适合在社群中交换、赠送。"
                    },
                    {
                        "name": "52TOYS 超活化 仕女",
                        "brand": "52TOYS",
                        "price_range": "100-300元",
                        "emotion_tags": ["话题", "交流"],
                        "reason": "设计有话题性，容易引发讨论。"
                    }
                ]
            },
            {
                "title": "收藏展示和全系列收集向",
                "types": ["收藏", "展示", "全制霸"],
                "why": "适合重视完整度、展示感和长期保值体验的人。",
                "trigger_words": ["收藏", "全套", "展示", "保值", "限量"],
                "risk_words": ["防潮", "积灰", "成本高", "超预算"],
                "products": [
                    {
                        "name": "泡泡玛特 MEGA 珍藏系列",
                        "brand": "泡泡玛特",
                        "price_range": ">1000元",
                        "emotion_tags": ["收藏", "全制霸"],
                        "reason": "大娃尺寸，收藏价值高。"
                    },
                    {
                        "name": "万代 METAL BUILD 高达",
                        "brand": "万代",
                        "price_range": ">1000元",
                        "emotion_tags": ["收藏", "保值"],
                        "reason": "合金材质，细节精致，升值潜力大。"
                    },
                    {
                        "name": "迪士尼 公主 典藏版人偶",
                        "brand": "迪士尼",
                        "price_range": "500-1000元",
                        "emotion_tags": ["集齐", "展示"],
                        "reason": "公主系列适合全套收集。"
                    }
                ]
            },
            {
                "title": "限定打卡和旅行纪念款",
                "types": ["限定", "园区", "打卡拍照"],
                "why": "适合喜欢到现场买限定、把旅行体验变成收藏的人。",
                "trigger_words": ["限定", "打卡", "园区", "地标", "拍照"],
                "risk_words": ["黄牛", "溢价", "排队", "冲动下单"],
                "products": [
                    {
                        "name": "迪士尼 五周年限定 城堡徽章",
                        "brand": "迪士尼",
                        "price_range": "100-300元",
                        "emotion_tags": ["限定", "打卡"],
                        "reason": "园区限定，打卡必备。"
                    },
                    {
                        "name": "泡泡玛特 城市限定系列",
                        "brand": "泡泡玛特",
                        "price_range": "<100元",
                        "emotion_tags": ["稀缺", "地标"],
                        "reason": "每个城市限定款，旅行打卡必买。"
                    },
                    {
                        "name": "草莓熊 限定 花束礼盒",
                        "brand": "迪士尼",
                        "price_range": "300-500元",
                        "emotion_tags": ["限定", "拍照"],
                        "reason": "节日限定款，适合拍照发圈。"
                    }
                ]
            }
        ]
    },

    "佛系水豚": {
        "tribe": "或许这里有你的同好：桌搭党、治愈系收集党、手感控",
        "brands_primary": ["52TOYS（摆件线）", "泡泡玛特（治愈系系列）", "TOP TOY（桌搭类）", "Jellycat", "Nici", "RoLife", "若态"],
        "channels": ["看口碑评价与细节图", "优先可退换渠道", "避免不明来源高溢价款"],
        "same_circle_buys": [
            {"name": "LIVHEART 恐龙抱枕", "brand": "LIVHEART", "price_range": "200-400元", "tag": "可抱可枕"},
            {"name": "迪士尼 松松 毛绒玩偶", "brand": "迪士尼", "price_range": "<100元", "tag": "小巧治愈"},
            {"name": "迪士尼 水晶球 音乐盒", "brand": "迪士尼", "price_range": "200-400元", "tag": "梦幻摆件"},
            {"name": "野兽派 熊猫 抱枕", "brand": "野兽派", "price_range": "200-300元", "tag": "软萌陪伴"},
            {"name": "Kinbor 手账本", "brand": "Kinbor", "price_range": "<100元", "tag": "安静独处"},
        ],
        "pitfalls": [
            "不要只看图片种草，注意实物色差、材质和清洁难度。",
            "治愈感很重要，但材质安全和耐用性更重要。"
        ],
        "items": [
            {
                "title": "治愈解压和柔软手感款",
                "types": ["治愈", "柔软", "解压"],
                "why": "适合压力大时想放松、偏好软乎乎和安全感的人。",
                "trigger_words": ["治愈", "放松", "柔软", "手感", "舒服"],
                "risk_words": ["掉毛", "异味", "材质不安全", "老化"],
                "products": [
                    {
                        "name": "Jellycat 邦尼兔 害羞系列",
                        "brand": "Jellycat",
                        "price_range": "100-300元",
                        "emotion_tags": ["治愈", "柔软"],
                        "reason": "触感极佳，公认的解压神器。"
                    },
                    {
                        "name": "Nici 绵羊 安抚玩偶",
                        "brand": "Nici",
                        "price_range": "100-300元",
                        "emotion_tags": ["放松", "安抚"],
                        "reason": "德国老牌，材质安全，抱感舒适。"
                    },
                    {
                        "name": "减压包子捏捏乐",
                        "brand": "杂牌",
                        "price_range": "<50元",
                        "emotion_tags": ["解压", "发泄"],
                        "reason": "便宜便携，随时捏一捏。"
                    }
                ]
            },
            {
                "title": "高颜值桌搭和审美摆件",
                "types": ["设计", "艺术感", "桌搭"],
                "why": "适合看重设计语言、摆在桌上也要好看的用户。",
                "trigger_words": ["设计", "颜值", "精致", "艺术感", "摆着好看"],
                "risk_words": ["色差", "积灰", "质感不符", "难打理"],
                "products": [
                    {
                        "name": "POP MART 小野 系列",
                        "brand": "泡泡玛特",
                        "price_range": "<100元",
                        "emotion_tags": ["设计", "艺术感"],
                        "reason": "小野风格独特，摆件感强。"
                    },
                    {
                        "name": "RoLife 仿真动物模型",
                        "brand": "RoLife",
                        "price_range": "100-300元",
                        "emotion_tags": ["写实", "精致"],
                        "reason": "细节逼真，适合作为桌面装饰。"
                    },
                    {
                        "name": "若态 八音盒 DIY小屋",
                        "brand": "若态",
                        "price_range": "300-500元",
                        "emotion_tags": ["手工", "美观"],
                        "reason": "拼装后非常精美，兼具功能性与观赏性。"
                    }
                ]
            },
            {
                "title": "陪伴感强、适合抱着和放在身边的产品",
                "types": ["陪伴", "抱抱", "温暖"],
                "why": "适合想要一点情绪支持、喜欢有存在感陪伴物的人。",
                "trigger_words": ["陪伴", "抱着", "治愈", "温暖", "不孤单"],
                "risk_words": ["难清洗", "依赖", "掉毛", "异味"],
                "products": [
                    {
                        "name": "IKEA 鲨鱼 毛绒玩具",
                        "brand": "宜家",
                        "price_range": "100-200元",
                        "emotion_tags": ["陪伴", "温暖"],
                        "reason": "网红鲨鱼，很多人抱着睡觉。"
                    },
                    {
                        "name": "Gund 泰迪熊 毛绒",
                        "brand": "Gund",
                        "price_range": "200-400元",
                        "emotion_tags": ["柔软", "陪伴"],
                        "reason": "可录音互动，增加陪伴感。"
                    },
                    {
                        "name": "长草颜文字 抱枕",
                        "brand": "名创优品",
                        "price_range": "<100元",
                        "emotion_tags": ["可爱", "抱抱"],
                        "reason": "表情治愈，适合抱着追剧。"
                    }
                ]
            },
            {
                "title": "适合安静独处和沉浸自赏的产品",
                "types": ["独处", "沉浸", "私密"],
                "why": "适合不爱晒图、偏好自己慢慢玩和静静欣赏的人。",
                "trigger_words": ["独处", "沉浸", "安静", "专注", "自己欣赏"],
                "risk_words": ["材料安全", "零件细小", "胶水", "占空间"],
                "products": [
                    {
                        "name": "DIY 数字油画",
                        "brand": "杂牌",
                        "price_range": "50-100元",
                        "emotion_tags": ["沉浸", "自我"],
                        "reason": "独自完成，享受过程，不分享。"
                    },
                    {
                        "name": "微缩模型 小屋 DIY",
                        "brand": "若态",
                        "price_range": "100-300元",
                        "emotion_tags": ["私密", "专注"],
                        "reason": "制作过程安静，成品自赏。"
                    },
                    {
                        "name": "单向历 设计款",
                        "brand": "单向空间",
                        "price_range": "<100元",
                        "emotion_tags": ["独处", "仪式"],
                        "reason": "每天撕一页，属于自己的时光。"
                    }
                ]
            }
        ]
    },

    "理智过客": {
        "tribe": "或许这里有你的同好：送礼党、轻度体验党、路人收藏",
        "brands_primary": ["迪士尼", "名创优品", "泡泡玛特", "乐高", "野兽派"],
        "channels": ["优先可退换", "避免高溢价冲动买", "送礼先确认对方偏好"],
        "same_circle_buys": [
            {"name": "乐高 花束 植物系列", "brand": "乐高", "price_range": "300-500元", "tag": "送礼稳妥"},
            {"name": "野兽派 香薰 礼盒", "brand": "野兽派", "price_range": "200-400元", "tag": "氛围感"},
            {"name": "酷乐潮玩 杂货 小摆件", "brand": "酷乐潮玩", "price_range": "<50元", "tag": "便宜有趣"},
            {"name": "官方旗舰店 预售款", "brand": "官方渠道", "price_range": "原价浮动", "tag": "可靠"},
            {"name": "迪士尼 米奇 经典公仔", "brand": "迪士尼", "price_range": "100-300元", "tag": "大众认知高"},
        ],
        "pitfalls": [
            "送礼别选太冷门的IP，优先大众认知度高的经典品牌。",
            "冲动消费前先搜同款比价，尽量保留小票。"
        ],
        "items": [
            {
                "title": "送礼稳妥、不容易出错的产品",
                "types": ["礼盒", "精致", "大众接受度高"],
                "why": "适合送朋友、家人、同事，重在稳妥和体面。",
                "trigger_words": ["礼物", "精致", "包装", "惊喜", "可爱"],
                "risk_words": ["冷门IP", "物流延误", "包装破损", "售后"],
                "products": [
                    {
                        "name": "迪士尼 公主 礼盒套装",
                        "brand": "迪士尼",
                        "price_range": "300-500元",
                        "emotion_tags": ["礼品", "精致"],
                        "reason": "包装精美，适合送礼。"
                    },
                    {
                        "name": "名创优品 三丽鸥 零食礼包",
                        "brand": "名创优品",
                        "price_range": "<100元",
                        "emotion_tags": ["实用", "可爱"],
                        "reason": "零食加玩偶，性价比高。"
                    },
                    {
                        "name": "泡泡玛特 限定盲盒礼盒",
                        "brand": "泡泡玛特",
                        "price_range": "100-300元",
                        "emotion_tags": ["惊喜", "礼物"],
                        "reason": "盲盒形式增加收礼人的惊喜感。"
                    }
                ]
            },
            {
                "title": "随手买一个也不心疼的平价小物",
                "types": ["平价", "小件", "即兴购买"],
                "why": "适合路过看到就买，重在轻松试水。",
                "trigger_words": ["随手", "可爱", "便宜", "小物件", "即兴"],
                "risk_words": ["被宰", "冲动消费", "不值", "退换麻烦"],
                "products": [
                    {
                        "name": "名创优品 盲袋",
                        "brand": "名创优品",
                        "price_range": "<50元",
                        "emotion_tags": ["随手", "平价"],
                        "reason": "逛街时看到随手拿一个，不心疼。"
                    },
                    {
                        "name": "泡泡玛特 自动售卖机 随机款",
                        "brand": "泡泡玛特",
                        "price_range": "<100元",
                        "emotion_tags": ["随机", "冲动"],
                        "reason": "路过机器顺手买一个，体验拆盒乐趣。"
                    },
                    {
                        "name": "迪士尼 钥匙扣 挂件",
                        "brand": "迪士尼",
                        "price_range": "50-100元",
                        "emotion_tags": ["可爱", "即兴"],
                        "reason": "小物件，适合偶遇时购买。"
                    }
                ]
            },
            {
                "title": "任务明确、按需求购买的产品",
                "types": ["指定款", "代购", "任务导向"],
                "why": "适合帮别人买、目标明确、不追求情绪体验的人。",
                "trigger_words": ["指定", "代购", "任务", "准确", "版本"],
                "risk_words": ["买错版本", "假货", "价格不透明", "等待过久"],
                "products": [
                    {
                        "name": "万代 高达 模型 指定款",
                        "brand": "万代",
                        "price_range": "100-300元",
                        "emotion_tags": ["任务", "准确"],
                        "reason": "帮朋友代购，指定型号，目标明确。"
                    },
                    {
                        "name": "泡泡玛特 指定盲盒系列",
                        "brand": "泡泡玛特",
                        "price_range": "<100元",
                        "emotion_tags": ["代购", "精确"],
                        "reason": "朋友指定要某个系列，直接买。"
                    },
                    {
                        "name": "限定潮玩预定",
                        "brand": "各类官方/代购渠道",
                        "price_range": "价格不定",
                        "emotion_tags": ["任务", "等待"],
                        "reason": "海外代购任务，需要等待。"
                    }
                ]
            },
            {
                "title": "只想了解圈子但不想深度入坑",
                "types": ["经典IP", "大众认知", "轻了解"],
                "why": "适合基本不怎么买，但想知道大家都在喜欢什么的人。",
                "trigger_words": ["看看", "了解", "经典", "大众", "轻度尝试"],
                "risk_words": ["硬消费", "跟风入坑", "买了不用"],
                "products": [
                    {
                        "name": "迪士尼 米奇 经典公仔",
                        "brand": "迪士尼",
                        "price_range": "100-300元",
                        "emotion_tags": ["大众熟知", "经典符号"],
                        "reason": "大众认知高，适合了解这个圈子。"
                    },
                    {
                        "name": "泡泡玛特 热门款展示",
                        "brand": "泡泡玛特",
                        "price_range": "<100元",
                        "emotion_tags": ["热门", "供了解"],
                        "reason": "不用重投入，也能知道当前流行什么。"
                    }
                ]
            }
        ]
    },
}

PERSONA_PROFILE = {
    "社交孔雀": {
        "title": "社交打卡型用户",
        "traits": ["爱分享、爱开箱", "追求新鲜感与话题", "容易被联名/限定刺激"],
        "pay_items": ["盲盒/抽选", "联名限定周边", "线下快闪体验"],
        "price": "100–500元/月（热度高时可能更高）",
        "buy_style": "更容易为热度、稀缺感和晒图价值买单",
        "advice_focus": "重点提醒理性看待溢价和黄牛炒价"
    },
    "圈层海王": {
        "title": "圈层归属型用户",
        "traits": ["围绕IP长期收集", "重视同好与归属", "更在意系列完整度"],
        "pay_items": ["IP周边/谷子", "棉花娃娃及配件", "社群活动/展会"],
        "price": "100–300元/月（随追更/补款波动）",
        "buy_style": "更容易为情怀、完整度和圈层连接持续消费",
        "advice_focus": "重点提醒预算控制和官方授权渠道"
    },
    "佛系水豚": {
        "title": "悦己治愈型用户",
        "traits": ["为自己解压治愈", "重视手感/颜值/质感", "不追热点，偏少而精"],
        "pay_items": ["治愈桌搭摆件", "毛绒/解压小物", "高口碑手办/摆件"],
        "price": "50–300元/月（更看重品质稳定）",
        "buy_style": "更容易为颜值、舒适感和长期陪伴感买单",
        "advice_focus": "重点提醒材质安全、色差和清洁成本"
    },
    "理智过客": {
        "title": "理性低涉入型用户",
        "traits": ["不常买，更多试水/送礼", "价格敏感，讨厌溢价", "优先可退换与口碑"],
        "pay_items": ["基础款试水", "礼赠向联名小件", "文具/生活小物"],
        "price": "≤100元/月（节日送礼时上升）",
        "buy_style": "更容易为实用性、稳妥感和低风险体验买单",
        "advice_focus": "重点提醒送礼选大众IP、冲动消费先比价"
    },
}

TEAM_INTRO = """
我们是一群对“成年人为何为童趣买单”充满好奇的大学生。  

通过 840 份问卷、30 场深度访谈和 1888 条网络评论，我们发现那些看似幼稚的玩具，其实是成年人在高压生活中为自己留出的温柔缝隙。  

正如一位受访者所说：“平时工作节奏太快，买个毛绒玩具就是给自己一点小小的放松。”  

童趣消费，本质上是一场温柔的自我养育——在理性现实的缝隙中，保留一点初心与热爱。  

我们希望通过这个测评工具，帮助更多人理解自己的消费心理，找到真正适合自己的情绪陪伴。
"""

ABOUT_US = """ 

**研究简介**  
我们是一群对“成年人为何为童趣买单”充满好奇的大学生。  
通过 840 份问卷、30 场深度访谈和 1888 条网络评论，我们发现，那些看似幼稚的玩具，其实是成年人在高压生活中为自己留出的温柔缝隙。  
正如一位受访者所说：“平时工作节奏太快，买个毛绒玩具就是给自己一点小小的放松。”  
童趣消费，本质上是一场温柔的自我养育——在理性现实的缝隙中，保留一点初心与热爱。

**核心数据可视化**  
- 双轨制参与比例：47.5%  
- 情感分析词云（正向 / 负向）  

**我们的初心故事**  
为什么研究成年人童趣消费？  
我们希望呼应“童趣是温柔的自我养育”的理念，让大家理解玩具背后的情绪价值。  
通过测评工具，帮助更多人理解自己的消费心理，找到真正适合自己的情绪陪伴。  

**荣誉墙 / 成果展示**  
- 采纳证明、表扬信、媒体报道
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
def price_range_matches(price_range: str, budget: int) -> bool:
    if price_range == "<50元":
        return budget < 50
    if price_range == "<100元":
        return budget < 100
    if price_range == "50-100元":
        return 50 <= budget <= 100
    if price_range == "100-200元":
        return 100 <= budget <= 200
    if price_range == "100-300元":
        return 100 <= budget <= 300
    if price_range == "200-400元":
        return 200 <= budget <= 400
    if price_range == "300-500元":
        return 300 <= budget <= 500
    if price_range == "500-1000元":
        return 500 <= budget <= 1000
    if price_range == ">1000元":
        return budget > 1000
    if price_range in ["价格不定", "原价浮动"]:
        return True
    return True


def filter_products_by_budget(products: list, budget: int) -> list:
    matched = [p for p in products if price_range_matches(p.get("price_range", ""), budget)]
    return matched if matched else products

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

            st.markdown("### 为你匹配的具体产品")
            matched_products = filter_products_by_budget(it.get("products", []), budget)
            if matched_products:
                for p in matched_products:
                    tags = "、".join(p.get("emotion_tags", []))
                    st.markdown(
                        f"""
                        <div class="mini-card">
                            <b>{p['name']}</b><br>
                            <span class="muted">品牌：{p['brand']} ｜ 价格：{p['price_range']}</span><br>
                            情绪标签：{tags}<br>
                            推荐理由：{p['reason']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("当前预算下暂无特别匹配的产品，已为你保留该方向的默认推荐。")

            st.markdown("### 同圈层人还买了")
            same_circle_buys = filter_products_by_budget(cfg.get("same_circle_buys", []), budget)
            if same_circle_buys:
                for p in same_circle_buys:
                    st.markdown(f"- {p['name']}（{p['brand']}，{p['price_range']}，{p['tag']}）")
            else:
                st.caption("当前预算下暂无同圈层热门产品。")

            st.markdown("### 避坑建议")
            for tip in cfg.get("pitfalls", []):
                st.markdown(f"- {tip}")

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
    # 顶部标题 + logo
    col_title, col_logo = st.columns([5, 1])  # 左大右小
    with col_title:
        st.markdown("<h2 style='margin-bottom:0px'>关于我们</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:10px'>{ABOUT_US}</div>", unsafe_allow_html=True)
    with col_logo:
        logo_path = IMG_DIR / "about" / "about_icon.jpg"
        if logo_path.exists():
            st.image(logo_path, width=240)  # 放大 logo

    st.markdown("---")

    # 核心数据可视化折叠框
    with st.expander("核心数据可视化（点击查看）"):
        st.markdown("- 双轨制参与比例：47.5%")
        st.markdown("- 情感分析词云（正向 / 负向）")

        cols = st.columns(3)
        double_track_path = IMG_DIR / "about" / "double_track.png"
        pos_wordcloud_path = IMG_DIR / "about" / "emotion_wordcloud_pos.png"
        neg_wordcloud_path = IMG_DIR / "about" / "emotion_wordcloud_neg.png"

        if double_track_path.exists():
            cols[0].image(double_track_path, caption="基础款 vs 限量款 受众交叉矩阵", width=380)
        if pos_wordcloud_path.exists():
            cols[1].image(pos_wordcloud_path, caption="正向情感词云", width=380)
        if neg_wordcloud_path.exists():
            cols[2].image(neg_wordcloud_path, caption="负向情感词云", width=380)

    st.markdown("---")

    # 荣誉墙 / 成果展示
    st.markdown("### 荣誉墙 / 成果展示")
    honor_imgs = [
        "honor_wall_1.png",
        "honor_wall_2.png",
        "honor_wall_3.png",
        "honor_wall_4.png",
    ]
    for img_file in honor_imgs:
        img_path = IMG_DIR / "about" / img_file
        if img_path.exists():
            st.image(img_path, width=750)
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
import os
import re
import glob
import jieba
from collections import Counter
import pandas as pd
from snownlp import SnowNLP
from snownlp import sentiment
from textblob import TextBlob
import numpy as np


# =========================
# 0) 可调参数
# =========================
DATA_DIR = r"C:\Users\27775\Desktop\正大杯\emotion\data"

# 文章类长文本阈值：>= 300 字 进入“文章/长文本”通道
LONG_TEXT_LEN = 300

# SnowNLP 的不确定区间：落在 [0.45, 0.55] 归为“不确定”
UNCERTAIN_LOW = 0.45
UNCERTAIN_HIGH = 0.55

# 去重：完全相同文本去重
DEDUP = True

# 输出文件名
OUT_XLSX = "ddd.xlsx"


# =========================
# 1) 读取 data 目录下所有 xlsx 的“评论文本”，合并成 reports(list[str])
# =========================
TEXT_COLS_PRIORITY = [
    "首次评价", "追加评价",
    "初评", "追评",
    "正文",
    "笔记内容",
    "内容", "标题",
]

BAN_COL_KEYWORDS = ["晒图", "图片", "视频", "网址", "链接", "fullpath", "realpath", "商家回复"]

def is_banned_col(colname: str) -> bool:
    col = str(colname)
    for kw in BAN_COL_KEYWORDS:
        if kw.lower() in col.lower():
            return True
    return False

def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"http[s]?://\S+", "", s)
    s = re.sub(r"www\.\S+", "", s)
    return s.strip()

reports = []
xlsx_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))

for fpath in xlsx_files:
    fname = os.path.basename(fpath)
    if fname.startswith("~$"):
        continue
    if fname.startswith("data_profile_"):
        continue

    try:
        xls = pd.ExcelFile(fpath, engine="openpyxl")
    except Exception as e:
        print(f"[SKIP] Cannot open {fname}: {e}")
        continue

    for sname in xls.sheet_names:
        try:
            df = pd.read_excel(fpath, sheet_name=sname, engine="openpyxl")
        except Exception as e:
            print(f"[SKIP] Read fail {fname} / {sname}: {e}")
            continue

        existing_text_cols = []
        for c in TEXT_COLS_PRIORITY:
            if c in df.columns and (not is_banned_col(c)):
                existing_text_cols.append(c)

        if not existing_text_cols:
            continue

        for col in existing_text_cols:
            for v in df[col].tolist():
                s = clean_text(v)
                if s:
                    reports.append(s)

print(f"[INFO] Collected raw texts: {len(reports)}")


# =========================
# 1.1) 去重（可选，但强烈推荐）
# =========================
dup_counter = None
if DEDUP:
    dup_counter = Counter(reports)
    reports = list(dup_counter.keys())
    print(f"[INFO] After dedup: {len(reports)} (removed {sum(dup_counter.values()) - len(reports)} duplicates)")
else:
    dup_counter = Counter(reports)  # 仍然统计一下次数，后面写入excel方便分析


# =========================
# 2) 训练情感分析模型（保持你的逻辑不变）
# =========================
# 注意：你训练+save 之后，并没有 load 回来用（SnowNLP 默认还是用自带模型）
# 这里不动你的逻辑，但建议你后续确认一下是否需要 sentiment.load(...)
sentiment.train('neg_all_dict.txt', 'pos_all_dict.txt')
sentiment.save('aaasentiment.marshal')
print("[INFO] Saved sentiment model to aaasentiment.marshal (not loaded by default SnowNLP)")


# =========================
# 2.1) Sanity check：固定句子测试 SnowNLP 是否“乱打分”
# =========================
sanity_texts = [
    "非常满意，质量很好，物流也很快！",
    "太差了，质量很差，完全不满意！",
    "一般般，还可以。",
    "气死我了，再也不买了。",
    "真的很喜欢，超出预期。",
]
print("\n===== [SANITY CHECK] SnowNLP sentiments =====")
for s in sanity_texts:
    try:
        sc = SnowNLP(s).sentiments
    except Exception as e:
        sc = None
        print(f"  [ERR] {s} -> {e}")
    print(f"  {sc:.4f}  | {s}" if sc is not None else f"  None   | {s}")
print("===== [SANITY CHECK END] =====\n")


# =========================
# 3) 情感分析：保留 SnowNLP>0.5
#    新增：长文本分流 + 不确定区间 + 去重次数写入 + TextBlob中文替代
# =========================
EMO_WORDS = set([
    "喜欢", "爱", "讨厌", "恶心", "开心", "快乐", "治愈", "解压", "感动", "惊喜", "失望",
    "生气", "愤怒", "难受", "崩溃", "烦", "焦虑", "舒服", "满意", "不满意", "后悔"
])

def chinese_subjectivity_fallback(text: str) -> float:
    """输出 0~1 主观性分数（启发式，不依赖英文模型）"""
    if not text:
        return 0.0

    punct = text.count("!") + text.count("！") + text.count("?") + text.count("？")
    punct_score = min(punct / 3.0, 1.0)

    first_person = sum(text.count(w) for w in ["我", "俺", "我们"])
    fp_score = min(first_person / 3.0, 1.0)

    tokens = [w for w in jieba.lcut(text) if w.strip()]
    if tokens:
        emo_cnt = sum(1 for w in tokens if w in EMO_WORDS)
        emo_score = min(emo_cnt / max(len(tokens), 1) * 5.0, 1.0)
    else:
        emo_score = 0.0

    parts = re.split(r"[。！？!?\n；;]+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        scores = [SnowNLP(p).sentiments for p in parts[:10]]
        var = float(np.std(scores))
        var_score = min(var * 3.0, 1.0)
    else:
        var_score = 0.0

    subj = 0.25 * punct_score + 0.25 * fp_score + 0.25 * emo_score + 0.25 * var_score
    return float(max(0.0, min(1.0, subj)))


# 四个“桶”：评论 vs 文章(长文本)，各自分为 正/负/不确定
short_pos, short_neg, short_unc = [], [], []
long_pos, long_neg, long_unc = [], [], []

def route_bucket(is_long: bool, label: str, item: dict):
    if not is_long:
        if label == "pos":
            short_pos.append(item)
        elif label == "neg":
            short_neg.append(item)
        else:
            short_unc.append(item)
    else:
        if label == "pos":
            long_pos.append(item)
        elif label == "neg":
            long_neg.append(item)
        else:
            long_unc.append(item)


for report in reports:
    report = report.strip()
    if not report:
        continue

    is_long = (len(report) >= LONG_TEXT_LEN)

    # SnowNLP 极性分析（0~1）
    try:
        polarity_snownlp = float(SnowNLP(report).sentiments)
    except Exception as e:
        # 极端情况下 SnowNLP 失败，直接跳过或标记
        polarity_snownlp = np.nan

    # TextBlob（对中文常返回0）
    blob = TextBlob(report)
    polarity_textblob = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    # 当 TextBlob 输出接近 0（无效）时：用 SnowNLP 映射到 -1~1
    if abs(polarity_textblob) < 1e-8 and (not np.isnan(polarity_snownlp)):
        polarity_textblob = polarity_snownlp * 2.0 - 1.0

    # 当主观性接近 0（无效）时：用中文启发式补
    if abs(subjectivity) < 1e-8:
        subjectivity = chinese_subjectivity_fallback(report)

    subjectivity_label = '主观' if subjectivity > 0.5 else '客观'

    # 去重次数（DEDUP=False 时也有意义；DEDUP=True 时用于保留原始频次信息）
    freq = int(dup_counter.get(report, 1)) if dup_counter is not None else 1

    result = {
        '评论': report,
        '文本长度': len(report),
        '出现次数': freq,
        '是否长文本': '是' if is_long else '否',
        'SnowNLP极性得分': polarity_snownlp,
        'TextBlob极性得分': polarity_textblob,
        '主观性得分': subjectivity,
        '主观性判断': subjectivity_label
    }

    # 不确定区间优先
    if (not np.isnan(polarity_snownlp)) and (UNCERTAIN_LOW <= polarity_snownlp <= UNCERTAIN_HIGH):
        route_bucket(is_long, "unc", result)
    else:
        # 保持你的逻辑：SnowNLP>0.5 为正，否则为负
        if (not np.isnan(polarity_snownlp)) and (polarity_snownlp > 0.5):
            route_bucket(is_long, "pos", result)
        else:
            route_bucket(is_long, "neg", result)


print("[INFO] Split summary:")
print(f"  Short texts (<{LONG_TEXT_LEN})  pos={len(short_pos)} neg={len(short_neg)} unc={len(short_unc)}")
print(f"  Long texts  (≥{LONG_TEXT_LEN})  pos={len(long_pos)}  neg={len(long_neg)}  unc={len(long_unc)}")


# =========================
# 4) 写入 Excel：新增 6 个 sheet（评论/文章 各自正负不确定）
# =========================
short_pos_df = pd.DataFrame(short_pos)
short_neg_df = pd.DataFrame(short_neg)
short_unc_df = pd.DataFrame(short_unc)

long_pos_df = pd.DataFrame(long_pos)
long_neg_df = pd.DataFrame(long_neg)
long_unc_df = pd.DataFrame(long_unc)

with pd.ExcelWriter(OUT_XLSX) as writer:
    # 短文本：更接近“评论”
    short_pos_df.to_excel(writer, sheet_name='评论_正向', index=False)
    short_neg_df.to_excel(writer, sheet_name='评论_负向', index=False)
    short_unc_df.to_excel(writer, sheet_name='评论_不确定', index=False)

    # 长文本：你说的“文章类”
    long_pos_df.to_excel(writer, sheet_name='文章_正向', index=False)
    long_neg_df.to_excel(writer, sheet_name='文章_负向', index=False)
    long_unc_df.to_excel(writer, sheet_name='文章_不确定', index=False)

print(f"[DONE] Saved to {OUT_XLSX}")
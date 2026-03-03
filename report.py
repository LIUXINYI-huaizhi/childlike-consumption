import os
import re
import glob
import jieba
from collections import Counter

import pandas as pd
import numpy as np
from snownlp import SnowNLP
from snownlp import sentiment
from textblob import TextBlob

import matplotlib.pyplot as plt
from wordcloud import WordCloud


# ============================================================
# 0) Config / Parameters
# ============================================================
DATA_DIR = r"C:\Users\27775\Desktop\正大杯\emotion\data"

LONG_TEXT_LEN = 300
UNCERTAIN_LOW = 0.45
UNCERTAIN_HIGH = 0.55
DEDUP = True

OUT_XLSX = os.path.join(DATA_DIR, "ddd.xlsx")
SUMMARY_XLSX = os.path.join(DATA_DIR, "text_sentiment_summary.xlsx")

PIE_SHORT = os.path.join(DATA_DIR, "pie_short_texts.png")
PIE_LONG = os.path.join(DATA_DIR, "pie_long_texts.png")

POS_TOP_WORDS_XLSX = os.path.join(DATA_DIR, "pos_top_words.xlsx")
NEG_TOP_WORDS_XLSX = os.path.join(DATA_DIR, "neg_top_words.xlsx")
POS_WORDCLOUD_PNG = os.path.join(DATA_DIR, "pos_wordcloud.png")
NEG_WORDCLOUD_PNG = os.path.join(DATA_DIR, "neg_wordcloud.png")

# Chinese font for WordCloud (required to render Chinese characters)
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"


# ============================================================
# 1) Read .xlsx files and extract text columns
#    NOTE: Your source Excel files use Chinese column names, so we keep those
#    strings in the column-name list. Everything else is English.
# ============================================================
TEXT_COLS_PRIORITY = [
    "首次评价", "追加评价",
    "初评", "追评",
    "正文",
    "笔记内容",
    "内容", "标题",
]

BANNED_COL_KEYWORDS = ["晒图", "图片", "视频", "网址", "链接", "fullpath", "realpath", "商家回复"]


def is_banned_column(colname: str) -> bool:
    col = str(colname)
    for kw in BANNED_COL_KEYWORDS:
        if kw.lower() in col.lower():
            return True
    return False


def clean_text(x) -> str:
    """Light cleaning: strip, collapse whitespace, remove URLs."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"http[s]?://\S+", "", s)
    s = re.sub(r"www\.\S+", "", s)
    return s.strip()


def load_texts_from_excels(data_dir: str) -> list[str]:
    texts: list[str] = []
    xlsx_files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))

    for fpath in xlsx_files:
        fname = os.path.basename(fpath)

        # Skip Excel temp files and any profiling/output files to avoid re-processing
        if fname.startswith("~$"):
            continue
        if fname.startswith("data_profile_"):
            continue
        if fname in ["ddd.xlsx", "text_sentiment_summary.xlsx", "pos_top_words.xlsx", "neg_top_words.xlsx"]:
            continue

        try:
            xls = pd.ExcelFile(fpath, engine="openpyxl")
        except Exception as e:
            print(f"[SKIP] Cannot open {fname}: {e}")
            continue

        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(fpath, sheet_name=sheet_name, engine="openpyxl")
            except Exception as e:
                print(f"[SKIP] Read failed {fname} / {sheet_name}: {e}")
                continue

            # Find available text columns on this sheet (by priority list)
            existing_text_cols = []
            for c in TEXT_COLS_PRIORITY:
                if c in df.columns and (not is_banned_column(c)):
                    existing_text_cols.append(c)

            if not existing_text_cols:
                continue

            # Collect cleaned non-empty texts
            for col in existing_text_cols:
                for v in df[col].tolist():
                    s = clean_text(v)
                    if s:
                        texts.append(s)

    return texts


# ============================================================
# 2) Train SnowNLP sentiment model (using your local neg/pos files)
# ============================================================
def train_snownlp_sentiment_model():
    sentiment.train("neg_all_dict.txt", "pos_all_dict.txt")
    sentiment.save("aaasentiment.marshal")
    print("[INFO] Trained & saved SnowNLP sentiment model -> aaasentiment.marshal")
    # NOTE: SnowNLP may not automatically load this model for SnowNLP(text).sentiments.
    # If you want to force using it, you may need sentiment.load("aaasentiment.marshal") depending on your SnowNLP version.


# ============================================================
# 3) Chinese subjectivity fallback (TextBlob often returns 0 for Chinese)
# ============================================================
EMOTION_WORDS = set([
    "喜欢", "爱", "讨厌", "恶心", "开心", "快乐", "治愈", "解压", "感动", "惊喜", "失望",
    "生气", "愤怒", "难受", "崩溃", "烦", "焦虑", "舒服", "满意", "不满意", "后悔"
])


def chinese_subjectivity_fallback(text: str) -> float:
    """Heuristic subjectivity score in [0, 1] for Chinese text."""
    if not text:
        return 0.0

    # Punctuation intensity
    punct = text.count("!") + text.count("！") + text.count("?") + text.count("？")
    punct_score = min(punct / 3.0, 1.0)

    # First-person usage
    first_person = sum(text.count(w) for w in ["我", "俺", "我们"])
    fp_score = min(first_person / 3.0, 1.0)

    # Emotion word density
    tokens = [w for w in jieba.lcut(text) if w.strip()]
    if tokens:
        emo_cnt = sum(1 for w in tokens if w in EMOTION_WORDS)
        emo_score = min(emo_cnt / max(len(tokens), 1) * 5.0, 1.0)
    else:
        emo_score = 0.0

    # Sentiment variability across sentences (more swings -> more subjective)
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


# ============================================================
# 4) Sentiment bucketing: short/long × pos/neg/uncertain
# ============================================================
def split_into_buckets(texts: list[str], freq_counter: Counter | None):
    short_pos, short_neg, short_unc = [], [], []
    long_pos, long_neg, long_unc = [], [], []

    def route(is_long: bool, label: str, item: dict):
        if not is_long:
            (short_pos if label == "pos" else short_neg if label == "neg" else short_unc).append(item)
        else:
            (long_pos if label == "pos" else long_neg if label == "neg" else long_unc).append(item)

    for text in texts:
        text = text.strip()
        if not text:
            continue

        is_long = (len(text) >= LONG_TEXT_LEN)

        try:
            sn_score = float(SnowNLP(text).sentiments)  # typically 0..1
        except Exception:
            sn_score = np.nan

        blob = TextBlob(text)
        tb_polarity = float(blob.sentiment.polarity)         # typically -1..1 (not reliable for Chinese)
        tb_subjectivity = float(blob.sentiment.subjectivity) # typically 0..1 (often 0 for Chinese)

        # If TextBlob polarity is ~0, map SnowNLP (0..1) to (-1..1) as a fallback
        if abs(tb_polarity) < 1e-8 and (not np.isnan(sn_score)):
            tb_polarity = sn_score * 2.0 - 1.0

        # If TextBlob subjectivity is ~0, use heuristic subjectivity for Chinese
        if abs(tb_subjectivity) < 1e-8:
            tb_subjectivity = chinese_subjectivity_fallback(text)

        subjectivity_label = "Subjective" if tb_subjectivity > 0.5 else "Objective"
        freq = int(freq_counter.get(text, 1)) if freq_counter is not None else 1

        item = {
            "Text": text,
            "TextLength": len(text),
            "Frequency": freq,
            "IsLongText": "Yes" if is_long else "No",
            "SnowNLPSentiment": sn_score,
            "TextBlobPolarity": tb_polarity,
            "TextBlobSubjectivity": tb_subjectivity,
            "SubjectivityLabel": subjectivity_label,
        }

        # Uncertain bucket by SnowNLP range
        if (not np.isnan(sn_score)) and (UNCERTAIN_LOW <= sn_score <= UNCERTAIN_HIGH):
            route(is_long, "unc", item)
        else:
            if (not np.isnan(sn_score)) and (sn_score > 0.5):
                route(is_long, "pos", item)
            else:
                route(is_long, "neg", item)

    return short_pos, short_neg, short_unc, long_pos, long_neg, long_unc


# ============================================================
# 5) Save ddd.xlsx with 6 English sheet names
# ============================================================
def save_results_excel(short_pos, short_neg, short_unc, long_pos, long_neg, long_unc):
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(short_pos).to_excel(writer, sheet_name="Short_Positive", index=False)
        pd.DataFrame(short_neg).to_excel(writer, sheet_name="Short_Negative", index=False)
        pd.DataFrame(short_unc).to_excel(writer, sheet_name="Short_Uncertain", index=False)

        pd.DataFrame(long_pos).to_excel(writer, sheet_name="Long_Positive", index=False)
        pd.DataFrame(long_neg).to_excel(writer, sheet_name="Long_Negative", index=False)
        pd.DataFrame(long_unc).to_excel(writer, sheet_name="Long_Uncertain", index=False)

    print(f"[DONE] Saved: {OUT_XLSX}")


# ============================================================
# 6) Summary table + pie charts
# ============================================================
def make_summary_and_pies():
    sheets = ["Short_Positive", "Short_Negative", "Short_Uncertain",
              "Long_Positive", "Long_Negative", "Long_Uncertain"]

    rows = []
    for sh in sheets:
        df = pd.read_excel(OUT_XLSX, sheet_name=sh)
        n_unique = len(df)
        n_weighted = int(df["Frequency"].sum()) if ("Frequency" in df.columns and n_unique) else n_unique
        mean_sn = float(df["SnowNLPSentiment"].mean()) if n_unique else np.nan
        rows.append([sh, n_unique, n_weighted, mean_sn])

    summary = pd.DataFrame(rows, columns=["Bucket", "UniqueCount", "WeightedCount", "MeanSnowNLP"])
    summary["Share_Unique"] = summary["UniqueCount"] / summary["UniqueCount"].sum()
    summary["Share_Weighted"] = summary["WeightedCount"] / summary["WeightedCount"].sum()
    summary.to_excel(SUMMARY_XLSX, index=False)
    print(f"[DONE] Saved: {SUMMARY_XLSX}")

    def pie_for(prefix: str, out_png: str, title: str):
        sub = summary[summary["Bucket"].str.startswith(prefix)].copy()
        labels = [x.replace(prefix + "_", "") for x in sub["Bucket"].tolist()]
        values = sub["WeightedCount"].tolist()

        plt.figure()
        plt.pie(values, labels=labels, autopct="%1.1f%%")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[DONE] Saved: {out_png}")

    pie_for("Short", PIE_SHORT, "Sentiment Distribution (Short Texts, Frequency-Weighted)")
    pie_for("Long", PIE_LONG, "Sentiment Distribution (Long Texts, Frequency-Weighted)")


# ============================================================
# 7) Top words & word clouds (use Short_Positive / Short_Negative)
# ============================================================
def build_wordclouds():
    pos_df = pd.read_excel(OUT_XLSX, sheet_name="Short_Positive")
    neg_df = pd.read_excel(OUT_XLSX, sheet_name="Short_Negative")

    # Minimal Chinese stopwords list (you can expand this)
    stopwords = set([
        "的", "了", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们",
        "这个", "那个", "非常", "真的", "一个", "感觉", "就是", "还有", "没有",
        "但是", "因为", "所以"
    ])

    def counter_from_df(df: pd.DataFrame) -> Counter:
        c = Counter()
        if df is None or len(df) == 0:
            return c
        for text, freq in zip(df["Text"].astype(str), df["Frequency"].astype(int)):
            for w in jieba.lcut(text):
                w = w.strip()
                if not w or w in stopwords:
                    continue
                if len(w) == 1:
                    continue
                # Frequency-weighted counts: closer to "public opinion heat"
                c[w] += freq
        return c

    pos_cnt = counter_from_df(pos_df)
    neg_cnt = counter_from_df(neg_df)

    pd.DataFrame(pos_cnt.most_common(80), columns=["Word", "Count"]).to_excel(POS_TOP_WORDS_XLSX, index=False)
    pd.DataFrame(neg_cnt.most_common(80), columns=["Word", "Count"]).to_excel(NEG_TOP_WORDS_XLSX, index=False)
    print(f"[DONE] Saved: {POS_TOP_WORDS_XLSX}")
    print(f"[DONE] Saved: {NEG_TOP_WORDS_XLSX}")

    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH} (please set FONT_PATH to a valid Chinese font file)")

    if pos_cnt:
        WordCloud(font_path=FONT_PATH, width=1400, height=900, background_color="white") \
            .generate_from_frequencies(dict(pos_cnt)) \
            .to_file(POS_WORDCLOUD_PNG)
        print(f"[DONE] Saved: {POS_WORDCLOUD_PNG}")

    if neg_cnt:
        WordCloud(font_path=FONT_PATH, width=1400, height=900, background_color="white") \
            .generate_from_frequencies(dict(neg_cnt)) \
            .to_file(NEG_WORDCLOUD_PNG)
        print(f"[DONE] Saved: {NEG_WORDCLOUD_PNG}")


# ============================================================
# Main
# ============================================================
def main():
    texts = load_texts_from_excels(DATA_DIR)
    print(f"[INFO] Collected raw texts: {len(texts)}")

    freq_counter = Counter(texts)
    if DEDUP:
        texts = list(freq_counter.keys())
        removed = sum(freq_counter.values()) - len(texts)
        print(f"[INFO] After dedup: {len(texts)} (removed {removed} duplicates)")

    # Train model (optional). If you're worried it doesn't take effect, you can skip this and use SnowNLP default model.
    train_snownlp_sentiment_model()

    # Bucket texts
    short_pos, short_neg, short_unc, long_pos, long_neg, long_unc = split_into_buckets(texts, freq_counter)

    print("[INFO] Bucket sizes:")
    print(f"  Short (<{LONG_TEXT_LEN})  pos={len(short_pos)} neg={len(short_neg)} unc={len(short_unc)}")
    print(f"  Long  (≥{LONG_TEXT_LEN})  pos={len(long_pos)}  neg={len(long_neg)}  unc={len(long_unc)}")

    # Save detailed results
    save_results_excel(short_pos, short_neg, short_unc, long_pos, long_neg, long_unc)

    # Summary + pie charts
    make_summary_and_pies()

    # Top words + word clouds
    build_wordclouds()

    print("[ALL DONE] All tables and figures have been generated under the data directory.")


if __name__ == "__main__":
    main()
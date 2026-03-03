import os
import re
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

# =========================
# Config
# =========================
DATA_DIR = r"./data"      # 改成你的data目录路径，比如 r"E:\xxx\data"
SAMPLE_N = 5              # 每列展示几个示例值
HEAD_N = 10               # 每个sheet抽样保存前多少行到samples文件
OUTPUT_PREFIX = "data_profile"

# 常见“文本列/评论列”关键词（中英文都考虑）
TEXT_COL_KEYWORDS = [
    "评论", "评价", "内容", "正文", "文本", "发言", "弹幕", "留言", "回复", "转发", "标题", "话题",
    "comment", "review", "content", "text", "message", "reply", "title"
]

# =========================
# Utils
# =========================
def safe_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def guess_text_columns(df: pd.DataFrame, top_k=6):
    """根据列名关键词 + 平均文本长度 + 非空比例，猜测文本列（适合后续情感分析）。"""
    scores = []
    for col in df.columns:
        ser = df[col]
        nonnull = ser.dropna()
        nonnull_ratio = len(nonnull) / max(len(df), 1)

        # 名称命中
        name_hit = 0
        col_lower = str(col).lower()
        for kw in TEXT_COL_KEYWORDS:
            if kw.lower() in col_lower:
                name_hit = 1
                break

        # 类型与长度特征
        if len(nonnull) == 0:
            avg_len = 0.0
            str_ratio = 0.0
        else:
            # 转字符串后看长度
            strs = nonnull.astype(str).map(safe_str)
            lens = strs.map(len)
            avg_len = float(lens.mean()) if len(lens) else 0.0
            # 有多少看起来像自然语言（长度>=6）
            str_ratio = float((lens >= 6).mean()) if len(lens) else 0.0

        # 综合打分（你可以按需求调权重）
        score = (
            2.0 * name_hit +
            1.5 * nonnull_ratio +
            1.0 * min(avg_len / 50.0, 2.0) +   # 平均长度越长越可能是文本
            1.0 * str_ratio
        )

        scores.append((col, score, name_hit, nonnull_ratio, avg_len, str_ratio))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k], scores  # 返回 top_k 以及全量明细

def column_profile(df: pd.DataFrame, col: str):
    ser = df[col]
    nonnull = ser.dropna()
    n = len(df)
    nn = len(nonnull)
    miss = n - nn
    miss_rate = miss / max(n, 1)

    # 取一些示例值（去重后前N个）
    examples = []
    if nn > 0:
        vals = nonnull.astype(str).map(safe_str)
        for v in vals.unique():
            if v == "":
                continue
            examples.append(v[:120])
            if len(examples) >= SAMPLE_N:
                break

    dtype = str(ser.dtype)
    nunique = int(nonnull.nunique()) if nn > 0 else 0

    # 判断是否数值列
    num_stats = {}
    if pd.api.types.is_numeric_dtype(ser):
        num_stats = {
            "mean": float(pd.to_numeric(ser, errors="coerce").mean()),
            "std": float(pd.to_numeric(ser, errors="coerce").std()),
            "min": float(pd.to_numeric(ser, errors="coerce").min()),
            "max": float(pd.to_numeric(ser, errors="coerce").max()),
        }

    return {
        "col": col,
        "dtype": dtype,
        "n_rows": n,
        "n_nonnull": nn,
        "missing": miss,
        "missing_rate": miss_rate,
        "nunique": nunique,
        "examples": " | ".join(examples),
        "num_stats_json": json.dumps(num_stats, ensure_ascii=False) if num_stats else "",
    }

# =========================
# Main
# =========================
def main():
    t0 = datetime.now()

    xlsx_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx found in: {DATA_DIR}")

    summary_rows = []
    colprof_rows = []
    sample_sheets = []  # (name, df_head)

    report_lines = []
    report_lines.append(f"DATA PROFILE REPORT")
    report_lines.append(f"Time: {t0.isoformat(timespec='seconds')}")
    report_lines.append(f"Data dir: {os.path.abspath(DATA_DIR)}")
    report_lines.append(f"Files: {len(xlsx_files)}")
    report_lines.append("=" * 80)

    for fpath in xlsx_files:
        fname = os.path.basename(fpath)

        try:
            xls = pd.ExcelFile(fpath, engine="openpyxl")
            sheet_names = xls.sheet_names
        except Exception as e:
            report_lines.append(f"[ERROR] Cannot open {fname}: {repr(e)}")
            continue

        report_lines.append(f"\nFILE: {fname}")
        report_lines.append(f"  Sheets: {sheet_names}")

        for sname in sheet_names:
            try:
                df = pd.read_excel(fpath, sheet_name=sname, engine="openpyxl")
            except Exception as e:
                report_lines.append(f"  [ERROR] Read sheet '{sname}' failed: {repr(e)}")
                continue

            n_rows, n_cols = df.shape
            cols = list(df.columns)

            # 猜测文本列
            top_text_cols, all_text_scores = guess_text_columns(df, top_k=6)

            summary_rows.append({
                "file": fname,
                "sheet": sname,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "columns": ", ".join([str(c) for c in cols[:80]]) + (" ..." if len(cols) > 80 else ""),
                "top_text_cols": ", ".join([f"{c}({score:.2f})" for c, score, *_ in top_text_cols]),
            })

            # 每列画像
            for col in cols:
                cp = column_profile(df, col)
                cp.update({"file": fname, "sheet": sname})
                colprof_rows.append(cp)

            # 保存前N行 sample
            head_df = df.head(HEAD_N).copy()
            # sheet名太长会影响Excel写入，这里做截断
            sheet_id = f"{fname}__{sname}"
            sheet_id = re.sub(r"[\[\]\:\*\?\/\\]", "_", sheet_id)
            sheet_id = sheet_id[:31]  # Excel sheet name max length
            sample_sheets.append((sheet_id, head_df))

            # 写可读报告
            report_lines.append(f"  SHEET: {sname}")
            report_lines.append(f"    Shape: {n_rows} rows x {n_cols} cols")
            report_lines.append(f"    Columns: {cols}")
            report_lines.append(f"    Top text-like cols:")
            for c, score, name_hit, nonnull_ratio, avg_len, str_ratio in top_text_cols:
                report_lines.append(
                    f"      - {c}: score={score:.2f}, name_hit={name_hit}, nonnull={nonnull_ratio:.2f}, "
                    f"avg_len={avg_len:.1f}, long_text_ratio={str_ratio:.2f}"
                )

    # 输出文件
    out_summary_xlsx = os.path.join(DATA_DIR, f"{OUTPUT_PREFIX}_summary.xlsx")
    out_samples_xlsx = os.path.join(DATA_DIR, f"{OUTPUT_PREFIX}_samples.xlsx")
    out_report_txt = os.path.join(DATA_DIR, f"{OUTPUT_PREFIX}_report.txt")

    summary_df = pd.DataFrame(summary_rows)
    colprof_df = pd.DataFrame(colprof_rows)

    # 写总览：summary + col_profile 两个sheet
    with pd.ExcelWriter(out_summary_xlsx, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="sheet_summary", index=False)
        colprof_df.to_excel(w, sheet_name="column_profile", index=False)

    # 写样本：每个sheet一个tab（可能很多，文件会大一些）
    with pd.ExcelWriter(out_samples_xlsx, engine="openpyxl") as w:
        for sheet_id, head_df in sample_sheets:
            head_df.to_excel(w, sheet_name=sheet_id, index=False)

    # 写文本报告
    with open(out_report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    t1 = datetime.now()
    print("Done.")
    print("Summary file:", out_summary_xlsx)
    print("Samples file:", out_samples_xlsx)
    print("Text report :", out_report_txt)
    print("Elapsed:", t1 - t0)

if __name__ == "__main__":
    main()
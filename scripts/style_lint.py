#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""style_lint.py — 子曰文风体检器

对一段文本做 16 项量化体检，与从本人语料实测得出的基线区间比对，
输出偏离项。用于「写完自检」和「判断某段文字像不像我写的」。

用法：
    python style_lint.py --file a.txt b.md
    python style_lint.py --text "一段文字"
    python style_lint.py --file x.txt --json
    python style_lint.py --corpus          # 打印 references/corpus 下全部语料的指标（校准用）
    python style_lint.py --register        # 只打印「我」的写作登记表（给模型看的紧凑版）

退出码：0 未超阈值；1 存在硬伤（AI 味词 / emoji / Markdown 结构符号）。
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HAN = "\u4e00-\u9fa5"

# ---------------------------------------------------------------- 词表

# AI 味 / 公文味标记：他的语料里从不出现，一旦出现即判定为「不像」
AI_SMELL = [
    "综上所述", "总而言之", "总的来说", "值得注意的是",
    "首先，", "其次，", "最后，", "一方面", "另一方面", "不仅如此", "与此同时",
    "在当今", "随着社会", "随着科技", "不可否认", "毋庸置疑", "众所周知",
    "赋能", "抓手", "闭环", "落地生根", "深度融合", "显著提升", "至关重要",
    "让我们", "本文将", "笔者", "综上", "由此可见", "深入剖析",
    "层层递进", "维度上", "从某种意义上说", "可以说，",
]

# 文言虚词（他高频使用，是文白夹杂的量化抓手）
WENYAN = "之其亦乃遂矣焉哉兮故皆犹岂莫未而"

# 口语词（他写长文时也不避口语，这是「不装」的证据）
COLLOQUIAL = ["啥", "咋", "弄", "搞", "挺", "哎", "啊", "呢", "嘛", "呗", "玩意儿", "这玩意儿"]

# 比喻标记（他偏爱「好似/宛如/就像/如同/仿佛/一般」）
SIMILE = ["好似", "宛如", "就像", "如同", "仿佛", "一般", "好像", "好比"]

# 概率化措辞（他极少下绝对判断）
HEDGE = ["可能", "或许", "大概率", "大概", "似乎", "也许", "应该", "说不定", "差不多"]

# 隐私/不适合入语料的条目已在建库时剔除，此处不做过滤


def han_count(t: str) -> int:
    return len(re.findall("[" + HAN + "]", t))


def clean(text: str) -> str:
    """剥掉语料文件里的说明行与元数据行。"""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("【说明】") or s.startswith("#"):
            continue
        lines.append(s)
    return "\n".join(lines)


def split_sentences(text: str):
    """按句末标点与换行切句；换行在观察清单式文本里就是句界。"""
    raw = re.split(r"[。！？!?；;\n]+", text)
    out = []
    for s in raw:
        s = s.strip().strip("“”\"'（）()")
        if han_count(s) >= 2:
            out.append(s)
    return out


def pk(n: int, chars: int) -> float:
    return round(n * 1000.0 / chars, 2) if chars else 0.0


def count_any(text: str, words) -> int:
    return sum(text.count(w) for w in words)


def metrics(text: str) -> dict:
    text = clean(text)
    chars = han_count(text)
    sents = split_sentences(text)
    n = len(sents) or 1
    lens = [han_count(s) for s in sents]
    avg = round(sum(lens) / n, 2)
    sd = round(statistics.pstdev(lens), 2) if len(lens) > 1 else 0.0
    short = sum(1 for x in lens if x <= 12)
    long_ = sum(1 for x in lens if x >= 40)
    commas = text.count("，") + text.count("、")
    q = text.count("？") + text.count("?")
    idiom4 = len(re.findall(r"(?<![" + HAN + r"])[" + HAN + r"]{4}(?![" + HAN + r"])", text))
    first = count_any(text, ["我", "自己"])
    dash = text.count("——") + text.count("—")
    emoji = len(re.findall("[\U0001F300-\U0001FAFF\u2600-\u27BF]", text))
    md_lines = sum(1 for ln in text.splitlines() if re.match(r"^\s*(#{1,6}\s|[-*+]\s|\d+\.\s|>\s)", ln))
    smell = [w for w in AI_SMELL if w in text]
    return {
        "chars": chars,
        "sentences": len(sents),
        "avg_sent_len": avg,
        "sent_len_sd": sd,
        "short_ratio": round(short / n, 3),
        "long_ratio": round(long_ / n, 3),
        "comma_per_sent": round(commas / n, 2),
        "question_ratio": round(q / n, 3),
        "wenyian_pk": pk(count_any(text, list(WENYAN)), chars),
        "idiom4_pk": pk(idiom4, chars),
        "first_person_pk": pk(first, chars),
        "colloquial_pk": pk(count_any(text, COLLOQUIAL), chars),
        "simile_pk": pk(count_any(text, SIMILE), chars),
        "hedge_pk": pk(count_any(text, HEDGE), chars),
        "dash_pk": pk(dash, chars),
        # 标点硬伤项：作者全语料 0 次
        "ellipsis": text.count("……") + text.count("...") + text.count("…"),
        "book_title": text.count("《"),
        "exclaim": text.count("！") + text.count("!"),
        "comma_colon_ratio": round(commas / (text.count("。") or 1), 2),
        "ai_smell": smell,
        "emoji": emoji,
        "md_struct_lines": md_lines,
    }

# ---------------------------------------------------------------- 基线
# 由 references/corpus 实测得出（长随笔 / 短札 / 观察清单三档），
# lo/hi 为「宽松可接受区间」，超出记 WARN；硬伤项单独判定。
BASELINE = {
    # 实测（03_长随笔_寻探我自_节选，2513 字 / 109 句）：句长 23.06、短句 0.211、
    # 长句 0.128、逗号 1.92、文言 14.33、四字 14.33、第一人称 31.44、比喻 4.38、模糊 6.76
    "long_essay": {
        "avg_sent_len": (16.0, 34.0),
        "sent_len_sd": (10.0, 24.0),
        "short_ratio": (0.15, 0.35),
        "long_ratio": (0.08, 0.28),
        "comma_per_sent": (1.40, 2.60),
        "question_ratio": (0.02, 0.12),
        "wenyian_pk": (8.0, 28.0),
        "idiom4_pk": (8.0, 24.0),
        "first_person_pk": (18.0, 45.0),
        "colloquial_pk": (1.0, 12.0),
        "simile_pk": (1.5, 12.0),
        "hedge_pk": (3.0, 16.0),
        "dash_pk": (0.0, 3.0),
    },
    # 实测（01_观察清单 172 字 + 02_短札 1963 字）：句长 10.75~33.84、逗号 0.62~2.98、
    # 第一人称 0~16.81、问句率 0.052~0.25、比喻 0~0
    "note": {
        "avg_sent_len": (8.0, 38.0),
        "sent_len_sd": (5.0, 26.0),
        "short_ratio": (0.10, 0.75),
        "long_ratio": (0.0, 0.38),
        "comma_per_sent": (0.50, 3.20),
        "question_ratio": (0.03, 0.32),
        "wenyian_pk": (4.0, 24.0),
        "idiom4_pk": (0.0, 22.0),
        "first_person_pk": (0.0, 35.0),
        "colloquial_pk": (0.0, 24.0),
        "simile_pk": (0.0, 10.0),
        "hedge_pk": (0.0, 18.0),
        "dash_pk": (0.0, 3.0),
    },
}


def judge(m: dict, mode: str):
    """按字数缩放容差：短稿的比值指标方差大，边界外不判失败。

    < 800 字：容差 25%；800-2000 字：15%；> 2000 字：8%。
    硬伤（AI 味词 / emoji / Markdown 结构）不受容差影响。
    """
    base = BASELINE[mode]
    tol = 0.25 if m["chars"] < 800 else (0.15 if m["chars"] < 2000 else 0.08)
    rows, warns = [], []
    for k, (lo, hi) in base.items():
        v = m[k]
        lo2 = lo * (1 - tol)
        hi2 = hi * (1 + tol) if hi > 0 else 0.0
        ok = lo2 <= v <= hi2
        flag = "ok " if ok else ("LOW" if v < lo2 else "HIGH")
        rows.append((k, v, round(lo2, 2), round(hi2, 2), flag))
        if not ok:
            warns.append("%s=%s 超出 [%s, %s]（容差 %d%%）" % (k, v, round(lo2, 2), round(hi2, 2), tol * 100))
    hard = []
    if m["ai_smell"]:
        hard.append("AI 味词：" + "、".join(m["ai_smell"][:6]))
    if m["emoji"]:
        hard.append("含 %d 个 emoji" % m["emoji"])
    if m["md_struct_lines"]:
        hard.append("含 %d 行 Markdown 结构符号" % m["md_struct_lines"])
    if m["ellipsis"]:
        hard.append("含 %d 个省略号（本人全语料 0 次，改断句）" % m["ellipsis"])
    if m["book_title"]:
        hard.append("含 %d 处书名号（本人写「道德经」不加书名号）" % m["book_title"])
    return rows, warns, hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", nargs="*", default=[])
    ap.add_argument("--text")
    ap.add_argument("--mode", choices=["long_essay", "note", "auto"], default="auto")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent

    if a.corpus:
        for p in sorted((root / "references" / "corpus").glob("*.txt")):
            m = metrics(p.read_text(encoding="utf-8"))
            print(p.name, json.dumps(m, ensure_ascii=False))
        return 0

    if a.register:
        reg = {
            "句子": "均句长 23 汉字（长随笔）/ 34 汉字（短札）；短句≤12 字占 15-35%；长句≥40 字占 8-28%",
            "标点": "逗号密集（每句 1.4-2.6 个），一逗到底；破折号几乎不用；问号收段",
            "词法": "文白夹杂，四字格每千字约 14 个；文言虚词每千字 12-14 字",
            "人称": "第一人称每千字 31 次，「我」是叙述主轴，从不抽离为第三人称",
            "口气": "「可能/或许/大概」概率化措辞（每千字约 7 次），绝不写绝对判断",
            "比喻": "每千字约 4 个，取自自然与身体（蒲公英、尘埃、土地、苗、秧）",
            "禁止": ["综上所述", "首先/其次/最后", "值得注意的是", "赋能/抓手/闭环", "emoji", "Markdown 标题与项目符号"],
        }
        print(json.dumps(reg, ensure_ascii=False, indent=2))
        return 0

    if a.text:
        texts = [a.text]
    else:
        texts = [Path(f).read_text(encoding="utf-8") for f in a.file]
    text = "\n".join(texts)
    m = metrics(text)

    if a.json:
        print(json.dumps(m, ensure_ascii=False, indent=2))
        return 0

    mode = a.mode
    if mode == "auto":
        mode = "long_essay" if m["chars"] >= 600 else "note"
    rows, warns, hard = judge(m, mode)
    print("模式：%s   汉字数：%d   句数：%d" % (mode, m["chars"], m["sentences"]))
    for k, v, lo, hi, flag in rows:
        print("  %-16s %8s   基线[%s, %s]  %s" % (k, v, lo, hi, flag))
    print("  标点：省略号 %d ｜ 书名号 %d ｜ 感叹号 %d ｜ 逗号:句号 %.2f"
          % (m["ellipsis"], m["book_title"], m["exclaim"], m["comma_colon_ratio"]))
    print()
    if hard:
        print("[硬伤]")
        for h in hard:
            print("  ✗ " + h)
    if warns:
        print("[偏离]")
        for w in warns:
            print("  ! " + w)
    if not warns and not hard:
        print("PASS：全部指标落在本人基线内。")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())

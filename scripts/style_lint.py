#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""style_lint.py — 子曰文风「表层特征」检测器 v2

【能力边界，先读这段】
本脚本只测可以计量的表层特征：句长分布、标点习惯、词表命中、结构符号。
它无法判断一段文字读起来是否像本人，也无法判断思路是否连贯、用典是否恰当、
内容是否准确。

PASS 的含义仅限于：这些表层指标落在语料基线内。
不得把 PASS 当成「风格验证成功」——真正的验证需要人做盲测，见 EVALUATION.md。

【指标口径说明】
- idiom_pk        固定四字表达（封闭词表命中）/ 千字。
                  **词表只收四字条目**；词表外的、以及超过四字的固定短语
                  （如「力不能缚鸡」「野火烧不尽」）一律不计入。见 --check-lexicon。
- cluster4_pk     长度 ≥4 的连续汉字串 / 千字。这是「四字块密度」的粗代理，
                  包含普通词组与专名，**不代表成语使用量**。
- wo_pk           汉字「我」的出现频率。
- self_pk         「自己」的出现频率。
- first_uniq_pk   「我」+「自己」−「我自己/我们自己」，避免同指重复计数。
- wenyan_char_pk  文言候选字符频率（之其亦乃遂矣焉哉兮故皆犹岂莫未而）。
                  **这是字符频率，不是语法分析**；「而」「其」「之」在现代汉语里
                  同样常见（而且、其中、之后），所以该值高不等于「文白夹杂程度高」。
- classical_mark_pk 强文言标记字频率（矣焉哉兮乃遂亦岂莫犹），现代汉语少用，判别力更强。
- simile_pk       强比喻标记（好似/宛如/就像/如同/仿佛/好比/恰似）。
                  「好像」「一般」归入 ambiguous 提示，不计入比喻，因为
                  「一般人」「一般情况下」是普通用法。
- 其余词频（口语、模糊限定）同为候选词命中数，不是语法判断。

用法：
    python style_lint.py --file a.txt
    python style_lint.py --text "一段话"
    python style_lint.py --file a.txt --mode long_essay
    python style_lint.py --file a.txt --mode generic     # 非随笔文本，只提示不判定
    python style_lint.py --file a.txt --json
    python style_lint.py --corpus        # 打印 references/corpus 的指标（校准用）
    python style_lint.py --register      # 紧凑版写作登记表
    python style_lint.py --check-lexicon # 校验四字词表条目长度

退出码（与 README、SKILL.md、EVALUATION.md 保持一致）：
    0  通过：无硬伤，也无提示
    1  硬伤：**仅限该体裁下确定不该出现的元素**
    2  提示：硬伤之外的一切（标点偏好、套话、区间偏离、结构符号）

硬伤按体裁判定（v2.1 的关键改动）：
    long_essay   Markdown 结构符号、emoji、行话（赋能/抓手/闭环…）
    note         行话
    generic      无硬伤，全部降为提示

省略号、书名号、感叹号，以及「综上所述」「由此可见」「毋庸置疑」在任何体裁下
都只是**提示**——那是个人标点与用词偏好，不是所有中文文本的正确性标准。
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
HAN_RE = re.compile("[" + HAN + "]")
HAN_CHAR = re.compile("[" + HAN + "]")

# ---------------------------------------------------------------- 词表
# 说明：以下均为「候选词命中」用的封闭词表，不是语法标注结果。

# 行话级：空转表达。个人随笔/短札里出现即不像（generic 体裁降为提示）
JARGON_PHRASES = [
    "赋能", "抓手", "闭环", "落地生根", "深度融合", "全面提升",
]

# 提示级：在说明文/议论文里完全正常，只作提醒，绝不判定失败
SOFT_PHRASES = [
    "综上所述", "由此可见", "毋庸置疑",
    "首先，", "其次，", "最后，", "因此，", "总的来说，", "总而言之，",
    "值得注意的是，", "一方面，", "另一方面，",
    "不可否认", "众所周知", "显著提升", "至关重要", "层层递进", "深入剖析",
    "从某种意义上说", "让我们", "本文将", "笔者", "在当今",
    "随着社会", "随着科技",
]

# 固定四字表达（封闭词表，命中即计）。未登录的四字词组不会计入 idiom_pk。
IDIOMS = frozenset("""
一心一意 三心二意 四面八方 五湖四海 七上八下 九牛一毛 十全十美
不知不觉 不慌不忙 不紧不慢 不声不响 不言而喻 不约而同 不由自主
不闻不问 不屑一顾 不置可否 不欢而散 不甘示弱 不计其数
千方百计 千言万语 千辛万苦 万无一失 理所当然 理直气壮
自言自语 自由自在 自作自受 自相矛盾 自给自足 自暴自弃
大同小异 大起大落 大彻大悟 大惊小怪 大器晚成 大相径庭
小心翼翼 小题大做 小肚鸡肠
无中生有 无动于衷 无微不至 无所适从 无能为力 无济于事 无独有偶
有条不紊 有气无力 有始有终 有增无减 有恃无恐
半信半疑 半途而废 半推半就
各抒己见 各得其所 各行其是
同心协力 同舟共济
如释重负 如履薄冰 如鱼得水 如火如荼 如影随形
好高骛远 好逸恶劳
妄自菲薄 安居乐业
心平气和 心灰意冷 心猿意马 心甘情愿 心照不宣 心旷神怡 心外无物
念念不忘 恋恋不舍
持之以恒 按部就班 挥之不去
推心置腹 措手不及
望而生畏 朝三暮四 杞人忧天
格格不入 欲盖弥彰
水到渠成 水滴石穿
深思熟虑 沧海桑田
漫不经心 潜移默化
牵强附会 独善其身 琳琅满目
画蛇添足 目瞪口呆
相形见绌 眉飞色舞
矢志不渝 破釜沉舟
蔚然成风 融会贯通
见微知著 言简意赅
谨小慎微 责无旁贷
身临其境 车载斗量
迎刃而解 适得其反
锲而不舍 随心所欲
雷厉风行 顺理成章
风平浪静 风雨无阻
一目了然 一五一十 一清二楚 一针见血 一如既往 一意孤行 一败涂地
三言两语 三番五次 五颜六色 五花八门 十拿九稳
不三不四 不伦不类 不折不扣 不慌不忙
井井有条 井然有序
光明正大 光明磊落
兴高采烈 兴致勃勃
冷眼旁观 冷嘲热讽
刻舟求剑 削足适履
勇往直前 势如破竹
化险为夷 十全十美
坐井观天 坚如磐石
多此一举 夜郎自大
守株待兔 实事求是
完璧归赵 对症下药
层出不穷 层峦叠嶂
川流不息 巧夺天工
废寝忘食 异想天开
弱不禁风 强人所难
得不偿失 得心应手
循规蹈矩 微乎其微
心悦诚服 忠言逆耳
急功近利 怨天尤人
惟妙惟肖 惊心动魄
想方设法 四面八方
手忙脚乱 手舞足蹈
投机取巧 拨云见日
拾金不昧 挥洒自如
有条有理 望穿秋水
杯弓蛇影 柳暗花明
栩栩如生 模棱两可
欢天喜地 欲速不达
步步为营 死心塌地
沧海一粟 流水不腐
浑然天成 焦头烂额
独具匠心 独树一帜
痛定思痛 百无聊赖
百折不挠 相辅相成
瞬息万变 破镜重圆
硕果累累 神通广大
祸不单行 稍纵即逝
突飞猛进 立竿见影
精打细算 精雕细琢
经天纬地 络绎不绝
缺一不可 罄竹难书
耳濡目染 聚沙成塔
胆战心惊 背水一战
舍己为人 良莠不齐
草木皆兵 落井下石
虚怀若谷 蛛丝马迹
街谈巷议 触类旁通
言之凿凿 言外之意
诚惶诚恐 谈虎色变
负荆请罪 赏心悦目
走马观花 赴汤蹈火
踌躇满志 车载斗量
近水楼台 返璞归真
迫在眉睫 追根溯源
通宵达旦 造谣惑众
郑人买履 量力而行 金玉良言
铁证如山 锲而不舍
闭门造车 防微杜渐
阳奉阴违 随波逐流
隐入尘埃 集思广益
青出于蓝 顺水推舟
风起云涌 饱经风霜
马到成功 驾轻就熟
高瞻远瞩 鬼斧神工
鱼贯而入 鸟语花香
一曝十寒 不了了之 自欺欺人 悬而未决
半途而废 从容不迫 云卷云舒 素履以往
穷且益坚 生生不息 运筹帷幄 胸有成竹
壮士断腕 所向披靡 天马行空 垂垂老矣
肝胆气郁 礼崩乐坏 顺其自然 顺其本然
不以物喜 不以己悲 心外无物 一苇以航
""".split())

# 文言候选字符（字符频率，非语法判断）
WENYAN_CANDIDATE = "之其亦乃遂矣焉哉兮故皆犹岂莫未而"
# 强标记：现代汉语书面语中很少单独使用，判别力更强
WENYAN_STRONG = "矣焉哉兮乃遂亦岂莫犹"

COLLOQUIAL = ["啥", "咋", "弄", "搞", "挺", "哎", "啊", "呢", "嘛", "呗", "玩意儿"]

# 强比喻标记（明确引出喻体）
SIMILE_STRONG = ["好似", "宛如", "就像", "如同", "仿佛", "好比", "恰似", "似的"]
# 两可用法：可能是比喻，也可能是普通限定/推测，仅作提示不计入比喻
SIMILE_AMBIGUOUS = ["好像", "一般"]

HEDGE = ["可能", "或许", "大概率", "大概", "似乎", "也许", "应该", "说不定", "差不多"]


# ---------------------------------------------------------------- 基础函数

def han_count(t: str) -> int:
    return len(HAN_CHAR.findall(t))


def strip_corpus_notes(text: str) -> str:
    """只剥掉语料自带的【说明】行。

    **不删 `#` 开头的行**：Markdown 标题检测必须基于原始文本，
    否则结构化符号会在检查前被清掉，永远测不出来。
    """
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("【说明】"):
            continue
        lines.append(s)
    return "\n".join(lines)


def md_struct_lines(text: str) -> int:
    """在原始文本上统计 Markdown 结构行（标题 / 列表 / 引用）。"""
    return sum(
        1
        for ln in text.splitlines()
        if re.match(r"^\s*(#{1,6}\s|[-*+]\s|\d+\.\s|>\s)", ln)
    )


def split_sentences(text: str):
    """按句末标点与换行切句；换行在清单式文本里就是句界。"""
    out = []
    for s in re.split(r"[。！？!?；;\n]+", text):
        s = s.strip().strip("“”\"'（）()")
        if han_count(s) >= 2:
            out.append(s)
    return out


def pk(n: int, chars: int) -> float:
    return round(n * 1000.0 / chars, 2) if chars else 0.0


def count_any(text: str, words) -> int:
    return sum(text.count(w) for w in words)


def count_simile(text: str):
    """返回 (强比喻标记数, 两可用法数)。

    「一般人」「一般情况下」「一般而言」「一般说来」等为普通限定，不计入。
    """
    strong = count_any(text, SIMILE_STRONG)
    amb = 0
    for w in SIMILE_AMBIGUOUS:
        if w != "一般":
            amb += text.count(w)
            continue
        for m in re.finditer("一般", text):
            nxt = text[m.end():m.end() + 2]
            if not re.match(r"(?:人|情况|而言|说来|地|般|性|化|都|是|不|无二|来讲|来说)", nxt):
                amb += 1
    return strong, amb


def count_idioms(text: str) -> int:
    """固定四字表达命中数（贪心、不重叠）。

    与旧版的关键差异：**不再要求前后不是汉字**。
    「他做事悬而未决，让人着急」里「悬而未决」照样命中。
    """
    hits, i, n = 0, 0, len(text)
    while i + 4 <= n:
        if text[i:i + 4] in IDIOMS:
            hits += 1
            i += 4
        else:
            i += 1
    return hits


def count_long_clusters(text: str) -> int:
    """长度 ≥4 的连续汉字串个数（四字块密度的粗代理，非成语计数）。"""
    return len(re.findall("[" + HAN + "]{4,}", text))


def person_counts(text: str):
    """返回 (我, 自己, 同指重复数, 去重后第一人称指称数)。

    「我自己」「我们自己」在两个单项里各被数了一次，去重时扣回。
    """
    wo = text.count("我")
    self_ = text.count("自己")
    both = text.count("我自己") + text.count("我们自己")
    return wo, self_, both, wo + self_ - both


# ---------------------------------------------------------------- 指标

def metrics(text: str) -> dict:
    md_lines = md_struct_lines(text)      # 先算，不能受清洗影响
    t = strip_corpus_notes(text)

    chars = han_count(t)
    sents = split_sentences(t)
    n = len(sents) or 1
    lens = [han_count(s) for s in sents]
    avg = round(sum(lens) / n, 2)
    sd = round(statistics.pstdev(lens), 2) if len(lens) > 1 else 0.0
    short = sum(1 for x in lens if x <= 12)
    long_ = sum(1 for x in lens if x >= 40)
    commas = t.count("，") + t.count("、")
    period = t.count("。") or 1
    q = t.count("？") + t.count("?")

    wo, self_, both, uniq = person_counts(t)
    strong_simile, amb_simile = count_simile(t)

    return {
        "chars": chars,
        "sentences": len(sents),
        "avg_sent_len": avg,
        "sent_len_sd": sd,
        "short_ratio": round(short / n, 3),
        "long_ratio": round(long_ / n, 3),
        "comma_per_sent": round(commas / n, 2),
        "comma_period_ratio": round(commas / period, 2),
        "question_ratio": round(q / n, 3),
        # 词法（封闭词表命中，非语法分析）
        "idiom_pk": pk(count_idioms(t), chars),
        "cluster4_pk": pk(count_long_clusters(t), chars),
        "wenyan_char_pk": pk(count_any(t, list(WENYAN_CANDIDATE)), chars),
        "classical_mark_pk": pk(count_any(t, list(WENYAN_STRONG)), chars),
        "colloquial_pk": pk(count_any(t, COLLOQUIAL), chars),
        "simile_pk": pk(strong_simile, chars),
        "simile_ambiguous_pk": pk(amb_simile, chars),
        "hedge_pk": pk(count_any(t, HEDGE), chars),
        # 人称（分开统计，避免同指重复）
        "wo_pk": pk(wo, chars),
        "self_pk": pk(self_, chars),
        "first_uniq_pk": pk(uniq, chars),
        # 标点与结构
        "dash_pk": pk(t.count("——") + t.count("—"), chars),
        "ellipsis": t.count("……") + t.count("…") + t.count("..."),
        "book_title": t.count("《"),
        "exclaim": t.count("！") + t.count("!"),
        "emoji": len(re.findall("[\U0001F300-\U0001FAFF\u2600-\u27BF]", t)),
        "md_struct_lines": md_lines,
        # 套话分级（行话可在提醒层，语体性套话一律只提醒）
        "jargon_phrases": [w for w in JARGON_PHRASES if w in t],
        "soft_phrases": [w for w in SOFT_PHRASES if w in t],
    }


# ---------------------------------------------------------------- 基线
# 由 references/corpus 实测得出（--corpus 可复现）：
#   长随笔 2513 字：idiom 5.97｜cluster4 118.98｜文言候选 14.33｜强文言 2.39
#                   人称去重 31.04（我 28.65 + 自己 2.79）｜比喻 3.18｜模糊 6.76
#   短札 1963 字：  idiom 5.60｜cluster4 109.53｜文言候选 12.23｜强文言 0.51
#                   人称去重 16.81｜比喻 0.00｜模糊 3.06
#   清单 172 字：   idiom 0.00｜cluster4 127.91｜人称 0.00｜比喻 0.00
# 区间只作「参考带」，不作出厂配额；超出时记为偏离并交人工判断。
BASELINE = {
    "long_essay": {
        "avg_sent_len": (16.0, 34.0),
        "sent_len_sd": (10.0, 24.0),
        "short_ratio": (0.15, 0.35),
        "long_ratio": (0.08, 0.28),
        "comma_per_sent": (1.40, 2.60),
        "question_ratio": (0.02, 0.12),
        "idiom_pk": (3.0, 12.0),
        "wenyan_char_pk": (8.0, 28.0),
        "first_uniq_pk": (18.0, 45.0),
        "colloquial_pk": (1.0, 12.0),
        "simile_pk": (0.0, 10.0),
        "hedge_pk": (3.0, 16.0),
        "dash_pk": (0.0, 3.0),
    },
    "note": {
        "avg_sent_len": (8.0, 38.0),
        "sent_len_sd": (5.0, 26.0),
        "short_ratio": (0.10, 0.75),
        "long_ratio": (0.0, 0.38),
        "comma_per_sent": (0.50, 3.20),
        "question_ratio": (0.03, 0.32),
        "idiom_pk": (0.0, 12.0),
        "wenyan_char_pk": (4.0, 24.0),
        "first_uniq_pk": (0.0, 35.0),
        "colloquial_pk": (0.0, 24.0),
        "simile_pk": (0.0, 10.0),
        "hedge_pk": (0.0, 18.0),
        "dash_pk": (0.0, 3.0),
    },
}


def tolerance(chars: int) -> float:
    if chars < 800:
        return 0.25
    if chars < 2000:
        return 0.15
    return 0.08


MODES = ("long_essay", "note", "generic")


def judge(m: dict, mode: str):
    """返回 (rows, notices, hard)。

    notices = 提示/偏离，**永不导致退出码 1**（个人偏好不是正确性标准）。
    hard    = 该体裁下确定不该出现的元素，仅 long_essay / note 生效。
    """
    if mode not in MODES:
        raise ValueError("unknown mode: %s" % mode)

    rows, notices, hard = [], [], []

    # 1) 区间比对。generic 不做：没有该体裁语料，不能拿随笔基线去量别体。
    if mode != "generic":
        base = BASELINE[mode]
        tol = tolerance(m["chars"])
        for k, (lo, hi) in base.items():
            v = m[k]
            lo2 = round(lo * (1 - tol), 2)
            hi2 = round(hi * (1 + tol), 2) if hi > 0 else 0.0
            ok = lo2 <= v <= hi2
            rows.append((k, v, lo2, hi2, "ok " if ok else ("LOW" if v < lo2 else "HIGH")))
            if not ok:
                notices.append("%s=%s 超出参考带 [%s, %s]（容差 %d%%）"
                               % (k, v, lo2, hi2, tol * 100))

    # 2) 行话：随笔/短札算硬伤，generic 降为提示
    if m["jargon_phrases"]:
        msg = "行话：" + "、".join(m["jargon_phrases"])
        (notices if mode == "generic" else hard).append(msg)

    # 3) 结构符号：只有长随笔算硬伤（他写长文是纯散文，不用任何标记）
    if m["md_struct_lines"]:
        msg = "含 %d 行 Markdown 结构符号（标题/列表/引用）" % m["md_struct_lines"]
        (hard if mode == "long_essay" else notices).append(msg)
    if m["emoji"]:
        msg = "含 %d 个 emoji" % m["emoji"]
        (hard if mode == "long_essay" else notices).append(msg)

    # 4) 标点偏好：一律只提示。个人标点习惯不等于文本错误。
    if m["ellipsis"]:
        notices.append("含 %d 个省略号（本人语料 0 次，属个人偏好，非错误）" % m["ellipsis"])
    if m["book_title"]:
        notices.append("含 %d 处书名号（本人写「道德经」不加书名号，属个人习惯，非错误）"
                       % m["book_title"])

    # 5) 语体性套话：一律只提示
    if m["soft_phrases"]:
        notices.append("语体性套话：" + "、".join(m["soft_phrases"])
                       + "（说明文可接受，随笔里偏模板）")

    return rows, notices, hard


def lexicon_audit():
    """返回 IDIOMS 中长度不等于 4 的条目（应为空）。"""
    return sorted({w for w in IDIOMS if len(w) != 4})


def evaluate(text: str, mode: str = "auto") -> dict:
    """供测试与外部调用的统一入口。"""
    m = metrics(text)
    if mode == "auto":
        mode = "long_essay" if m["chars"] >= 600 else "note"
    rows, notices, hard = judge(m, mode)
    code = 1 if hard else (2 if notices else 0)
    return {
        "mode": mode,
        "metrics": m,
        "rows": rows,
        "notices": notices,
        "hard": hard,
        "code": code,
        "verdict": {0: "PASS", 1: "HARD_FAIL", 2: "NOTICE"}[code],
    }


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="子曰文风表层特征检测器")
    ap.add_argument("--file", nargs="*", default=[])
    ap.add_argument("--text")
    ap.add_argument("--mode", choices=["long_essay", "note", "generic", "auto"], default="auto")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--check-lexicon", action="store_true")
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent

    if a.check_lexicon:
        bad = lexicon_audit()
        if bad:
            print("词表异常：以下条目不是四字，应移出 IDIOMS：")
            for w in bad:
                print("  - %s（%d 字）" % (w, len(w)))
            return 1
        print("词表 OK：%d 条，全部为四字。" % len(IDIOMS))
        return 0

    if a.corpus:
        d = root / "references" / "corpus"
        if not d.exists():
            print("未找到 references/corpus（该目录不随公开仓库分发）")
            return 2
        for p in sorted(d.glob("*.txt")):
            m = metrics(p.read_text(encoding="utf-8"))
            print(p.name, json.dumps(m, ensure_ascii=False))
        return 0

    if a.register:
        reg = {
            "核心原则（必须）": [
                "具体经验先行，判断在后",
                "个人视角，不替群体下结论",
                "事实用确定句，解释用推测句",
                "长短句随内容自然变化，不设配额",
                "允许真实的不知道",
            ],
            "可选特征（按需，不得为凑指标强行加）": [
                "「可是」单用转折", "文白夹杂与四字格", "用典后自嘲",
                "自然与身体意象", "问句收尾", "概率化措辞",
            ],
            "表层参考带（长随笔）": "句长 16-34｜逗号/句 1.4-2.6｜第一人称 18-45/千字｜固定四字 3-12/千字",
            "硬伤（仅 long_essay）": ["Markdown 标题与列表符号", "emoji", "行话：赋能/抓手/闭环/落地生根"],
            "硬伤（仅 note）": ["行话：赋能/抓手/闭环/落地生根"],
            "提示（所有体裁，不判失败）": [
                "省略号", "书名号", "综上所述/由此可见/毋庸置疑",
                "首先/其次/最后", "因此", "总的来说", "众所周知",
            ],
            "能力边界": "只测表层特征。PASS 不等于像，需人工盲测，见 EVALUATION.md",
        }
        print(json.dumps(reg, ensure_ascii=False, indent=2))
        return 0

    if a.text:
        texts = [a.text]
    elif a.file:
        texts = [Path(f).read_text(encoding="utf-8") for f in a.file]
    else:
        ap.error("需要 --file / --text / --corpus / --register 之一")

    text = "\n".join(texts)
    res = evaluate(text, a.mode)

    if a.json:
        out = {"mode": res["mode"], "verdict": res["verdict"], "code": res["code"],
               "metrics": res["metrics"], "notices": res["notices"], "hard": res["hard"]}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return res["code"]

    m = res["metrics"]
    print("模式：%s   汉字数：%d   句数：%d   判定：%s"
          % (res["mode"], m["chars"], m["sentences"], res["verdict"]))
    for k, v, lo, hi, flag in res["rows"]:
        print("  %-18s %8s   参考带[%s, %s]  %s" % (k, v, lo, hi, flag))
    print("  标点：省略号 %d ｜ 书名号 %d ｜ 感叹号 %d ｜ 逗号:句号 %.2f"
          % (m["ellipsis"], m["book_title"], m["exclaim"], m["comma_period_ratio"]))
    print("  人称：我 %.2f/千字 ｜ 自己 %.2f/千字 ｜ 去重后 %.2f/千字"
          % (m["wo_pk"], m["self_pk"], m["first_uniq_pk"]))
    print("  词法：固定四字 %.2f ｜ 四字块(粗) %.2f ｜ 文言候选字 %.2f ｜ 强文言标记 %.2f ｜ 比喻 %.2f"
          % (m["idiom_pk"], m["cluster4_pk"], m["wenyan_char_pk"],
             m["classical_mark_pk"], m["simile_pk"]))
    print()

    if res["hard"]:
        print("[硬伤]（退出码 1，仅该体裁适用）")
        for h in res["hard"]:
            print("  x " + h)
    if res["notices"]:
        print("[提示]（退出码 2，需人工判断，不是错误）")
        for w in res["notices"]:
            print("  ! " + w)
    if res["code"] == 0:
        print("PASS：表层指标全部落在参考带内，也无提示项。")
        print("注意：这只说明表层接近，不代表读起来像本人。见 EVALUATION.md")
    return res["code"]


if __name__ == "__main__":
    sys.exit(main())

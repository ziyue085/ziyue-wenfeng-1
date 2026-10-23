# -*- coding: utf-8 -*-
"""style_lint.py 的回归测试。

运行（在项目根目录）：
    python -m unittest discover -s tests -v

覆盖点：
1. 正常句子中的四字表达能被识别（旧版要求前后非汉字，会漏检）
2. 词表条目必须全是四字；超过四字的短语不计入
3. 「我自己」不造成第一人称重复计数（含明确数值断言）
4. 「一般人」不被认定为比喻
5. Markdown 标题不会被提前清理；是否判硬伤取决于体裁
6. 「首先/其次」「综上所述」等只提醒，不硬性失败
7. 不同长度 / 不同体裁采用对应标准
8. 退出码 0 / 1 / 2 与 verdict 精确对应
9. 边界输入不崩溃
10. 语料回归（若 references/corpus 存在）
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import style_lint as sl  # noqa: E402


class TestIdiom(unittest.TestCase):
    """1-2. 四字表达识别与词表约束"""

    def test_inline_idiom_is_detected(self):
        # 「悬而未决」前后都是汉字 + 标点混排，旧版正则漏检
        self.assertEqual(sl.count_idioms("他做事悬而未决，让人着急。"), 1)

    def test_idiom_in_middle_of_sentence(self):
        self.assertEqual(sl.count_idioms("这件事一直悬而未决，拖了很久。"), 1)

    def test_no_overlap_double_count(self):
        self.assertEqual(sl.count_idioms("不了了之"), 1)

    def test_non_idiom_quad_not_counted_as_idiom(self):
        self.assertEqual(sl.count_idioms("很大的树"), 0)

    def test_five_char_phrase_not_counted(self):
        # 「力不能缚鸡」是五字，词表不收，必须计 0
        self.assertEqual(sl.count_idioms("力不能缚鸡"), 0)

    def test_lexicon_all_four_chars(self):
        # 词表里不允许出现非四字条目（否则永远匹配不上，等于死条目）
        bad = sl.lexicon_audit()
        self.assertEqual(bad, [], msg="非四字条目：%s" % bad)

    def test_cluster_metric_is_labelled_separately(self):
        m = sl.metrics("这件事一直悬而未决，拖了很久。")
        self.assertEqual(m["idiom_pk"], round(1000.0 * 1 / 13, 2))
        self.assertNotEqual(m["cluster4_pk"], m["idiom_pk"])


class TestPerson(unittest.TestCase):
    """3. 第一人称口径（明确数值，不做自比较）"""

    def test_wo_self_not_double_counted(self):
        self.assertEqual(sl.person_counts("我自己知道"), (1, 1, 1, 1))

    def test_wo_men_self_not_double_counted(self):
        self.assertEqual(sl.person_counts("我们自己决定"), (1, 1, 1, 1))

    def test_plain_self_counts_once(self):
        self.assertEqual(sl.person_counts("管好你自己"), (0, 1, 0, 1))

    def test_metrics_values_are_exact(self):
        # 文本共 14 个汉字：我×1，自己×2，我自己×1 → 去重后 2
        text = "我自己想了很久，自己也想不明白。"
        self.assertEqual(sl.person_counts(text), (1, 2, 1, 2))
        m = sl.metrics(text)
        self.assertEqual(m["chars"], 14)
        self.assertEqual(m["wo_pk"], 71.43)
        self.assertEqual(m["self_pk"], 142.86)
        self.assertEqual(m["first_uniq_pk"], 142.86)


class TestSimile(unittest.TestCase):
    """4. 比喻误报"""

    def test_general_person_not_simile(self):
        self.assertEqual(sl.count_simile("一般人不会这么想。")[0], 0)

    def test_general_situation_not_simile(self):
        self.assertEqual(sl.count_simile("一般情况下他不会来。")[0], 0)

    def test_simile_marker_detected(self):
        self.assertEqual(sl.count_simile("他好似在等什么人。")[0], 1)

    def test_ambiguous_kept_out_of_simile(self):
        strong, amb = sl.count_simile("他好像在等什么人。")
        self.assertEqual(strong, 0)
        self.assertEqual(amb, 1)


class TestMarkdownAndPunctuation(unittest.TestCase):
    """5 + 新增：结构符号与标点偏好按体裁判定"""

    def test_heading_counted_before_clean(self):
        self.assertEqual(sl.md_struct_lines("# 标题\n正文第一句。"), 1)

    def test_heading_is_hard_only_in_long_essay(self):
        res = sl.evaluate("# 标题\n正文第一句。", mode="long_essay")
        self.assertTrue(any("Markdown" in h for h in res["hard"]))
        self.assertEqual(res["code"], 1)

    def test_heading_is_notice_in_note(self):
        res = sl.evaluate("# 标题\n正文第一句。", mode="note")
        self.assertEqual(res["hard"], [])
        self.assertTrue(any("Markdown" in n for n in res["notices"]))
        self.assertEqual(res["code"], 2)

    def test_bullet_counted(self):
        self.assertEqual(sl.md_struct_lines("- 第一条\n- 第二条"), 2)

    def test_corpus_note_line_removed_but_not_heading(self):
        text = "【说明】这是语料说明行\n正文一句。"
        self.assertNotIn("【说明】", sl.strip_corpus_notes(text))
        self.assertIn("#", sl.strip_corpus_notes("# 标题\n正文"))

    def test_ellipsis_is_never_hard(self):
        for mode in ("long_essay", "note", "generic"):
            res = sl.evaluate("我在等……", mode=mode)
            self.assertEqual(res["hard"], [], msg=mode)
            self.assertTrue(any("省略号" in n for n in res["notices"]), msg=mode)
            self.assertEqual(res["code"], 2, msg=mode)

    def test_book_title_is_never_hard(self):
        for mode in ("long_essay", "note", "generic"):
            res = sl.evaluate("我读了《道德经》。", mode=mode)
            self.assertEqual(res["hard"], [], msg=mode)
            self.assertTrue(any("书名号" in n for n in res["notices"]), msg=mode)

    def test_emoji_not_hard_in_note(self):
        res = sl.evaluate("今天挺开心的😀", mode="note")
        self.assertEqual(res["hard"], [])
        self.assertTrue(any("emoji" in n for n in res["notices"]))


class TestPhraseGrading(unittest.TestCase):
    """6. 套话分级：语体性表达只提醒，行话在随笔里才硬伤"""

    def test_first_second_is_notice_only(self):
        res = sl.evaluate("首先，我想说这件事。其次，还有别的。", mode="note")
        self.assertEqual(res["hard"], [])
        self.assertTrue(any("首先" in n for n in res["notices"]))
        self.assertEqual(res["code"], 2)

    def test_conclusion_phrases_are_notice_only(self):
        for word in ("综上所述", "由此可见", "毋庸置疑"):
            res = sl.evaluate("这是一句测试。" + word + "，方案可行。", mode="long_essay")
            self.assertEqual(res["hard"], [], msg=word)
            self.assertTrue(any(word in n for n in res["notices"]), msg=word)

    def test_jargon_is_hard_in_essay(self):
        res = sl.evaluate("我们要赋能业务，形成闭环。", mode="note")
        self.assertTrue(res["hard"])
        self.assertEqual(res["code"], 1)

    def test_jargon_is_notice_in_generic(self):
        res = sl.evaluate("我们要赋能业务，形成闭环。", mode="generic")
        self.assertEqual(res["hard"], [])
        self.assertEqual(res["code"], 2)


class TestModeSelection(unittest.TestCase):
    """7. 不同长度 / 体裁用不同标准"""

    def test_short_text_uses_note(self):
        self.assertEqual(sl.evaluate("今天去了学校。", mode="auto")["mode"], "note")

    def test_long_text_uses_long_essay(self):
        self.assertEqual(sl.evaluate("这是一段用来测试长度判定的句子。" * 60,
                                     mode="auto")["mode"], "long_essay")

    def test_explicit_mode_wins(self):
        self.assertEqual(sl.evaluate("今天去了学校。", mode="long_essay")["mode"], "long_essay")

    def test_generic_skips_range_comparison(self):
        res = sl.evaluate("这是一句普通说明文字。", mode="generic")
        self.assertEqual(res["rows"], [])

    def test_baselines_differ(self):
        self.assertNotEqual(sl.BASELINE["long_essay"], sl.BASELINE["note"])

    def test_tolerance_boundaries(self):
        self.assertEqual(sl.tolerance(0), 0.25)
        self.assertEqual(sl.tolerance(799), 0.25)
        self.assertEqual(sl.tolerance(800), 0.15)
        self.assertEqual(sl.tolerance(1999), 0.15)
        self.assertEqual(sl.tolerance(2000), 0.08)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            sl.judge(sl.metrics("一句话。"), mode="poetry")


class TestVerdicts(unittest.TestCase):
    """8. 退出码与 verdict 精确对应（0/1/2 全覆盖）"""

    PASS_TEXT = ("今天去了学校，路两边的树都黄了。风一吹，叶子就落下来。"
                 "这样的天气，人容易想起以前之事。我坐在那儿，心里没什么波澜。"
                 "是好是坏？我也不知道。")

    def test_code_0_pass(self):
        res = sl.evaluate(self.PASS_TEXT, mode="note")
        self.assertEqual(res["notices"], [], msg="不应有提示：%s" % res["notices"])
        self.assertEqual(res["hard"], [])
        self.assertEqual(res["code"], 0)
        self.assertEqual(res["verdict"], "PASS")

    def test_code_1_hard(self):
        res = sl.evaluate("我们要赋能业务。", mode="note")
        self.assertEqual(res["code"], 1)
        self.assertEqual(res["verdict"], "HARD_FAIL")

    def test_code_2_notice(self):
        res = sl.evaluate("我在等……", mode="note")
        self.assertEqual(res["code"], 2)
        self.assertEqual(res["verdict"], "NOTICE")

    def test_code_2_from_range_deviation(self):
        # 无硬伤、无提示性词，但句长指标明显偏离
        text = ("这是一段故意写得非常长而且没有任何短句停顿的文字用来触发篇幅指标偏离，"
                "同样这一句也保持着相当的长度并且不给出任何疑问语气，"
                "第三句仍然很长很平没有任何变化，第四句依旧如此继续堆叠下去。")
        res = sl.evaluate(text, mode="long_essay")
        self.assertEqual(res["hard"], [])
        self.assertEqual(res["code"], 2)
        self.assertEqual(res["verdict"], "NOTICE")

    def test_verdict_mapping_is_exact(self):
        self.assertEqual(sl.evaluate(self.PASS_TEXT, mode="note")["verdict"], "PASS")
        self.assertEqual(sl.evaluate("我们要赋能业务。", mode="note")["verdict"], "HARD_FAIL")
        self.assertEqual(sl.evaluate("我在等……", mode="note")["verdict"], "NOTICE")


class TestEdgeCases(unittest.TestCase):
    """9. 边界输入不崩溃"""

    def test_empty_text(self):
        res = sl.evaluate("", mode="note")
        self.assertEqual(res["metrics"]["chars"], 0)
        self.assertEqual(res["hard"], [])
        self.assertEqual(res["code"], 2)

    def test_punctuation_only(self):
        res = sl.evaluate("，。！？", mode="note")
        self.assertEqual(res["metrics"]["chars"], 0)

    def test_single_char(self):
        res = sl.evaluate("我", mode="note")
        self.assertEqual(res["metrics"]["chars"], 1)

    def test_no_punctuation_long_run(self):
        res = sl.evaluate("啊" * 200, mode="note")
        self.assertEqual(res["metrics"]["sentences"], 1)

    def test_ascii_text(self):
        res = sl.evaluate("hello world", mode="note")
        self.assertEqual(res["metrics"]["chars"], 0)


@unittest.skipUnless((ROOT / "references" / "corpus").exists(),
                     "references/corpus 不存在（不随公开仓库分发）")
class TestCorpusRegression(unittest.TestCase):
    """10. 语料回归：本人原文不应被判为硬伤"""

    def test_corpus_files_have_no_hard_issue(self):
        files = sorted((ROOT / "references" / "corpus").glob("*.txt"))
        self.assertGreaterEqual(len(files), 1)
        for p in files:
            res = sl.evaluate(p.read_text(encoding="utf-8"), mode="note")
            self.assertEqual(res["hard"], [], msg="%s 被误判硬伤：%s" % (p.name, res["hard"]))

    def test_corpus_long_essay_passes(self):
        p = ROOT / "references" / "corpus" / "03_长随笔_寻探我自_节选.txt"
        if not p.exists():
            self.skipTest("长随笔语料不存在")
        res = sl.evaluate(p.read_text(encoding="utf-8"), mode="long_essay")
        self.assertEqual(res["code"], 0, msg="偏离：%s" % res["notices"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

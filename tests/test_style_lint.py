# -*- coding: utf-8 -*-
"""style_lint.py 的回归测试。

运行（在项目根目录）：
    python -m unittest discover -s tests -v

覆盖点与需求一一对应：
1. 正常句子中的四字表达能被识别（旧版要求前后非汉字，会漏检）
2. 「我自己」不造成第一人称重复计数
3. 「一般人」不被认定为比喻
4. Markdown 标题不会被提前清理
5. 「首先/其次」不再无条件触发硬性失败（只提醒）
6. 不同长度 / 不同体裁采用对应标准
7. 退出码 0 / 1 / 2 与文档一致
8. 语料回归（若 references/corpus 存在）
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import style_lint as sl  # noqa: E402


class TestIdiom(unittest.TestCase):
    """1. 四字表达识别"""

    def test_inline_idiom_is_detected(self):
        # 「悬而未决」前后都是汉字 + 标点混排，旧版正则漏检
        text = "他做事悬而未决，让人着急。"
        self.assertGreaterEqual(sl.count_idioms(text), 1)

    def test_idiom_in_middle_of_sentence(self):
        text = "这件事一直悬而未决，拖了很久。"
        self.assertGreaterEqual(sl.count_idioms(text), 1)

    def test_no_overlap_double_count(self):
        # 「不了了之」命中后应跳过，不重复计
        self.assertEqual(sl.count_idioms("不了了之"), 1)

    def test_non_idiom_quad_not_counted_as_idiom(self):
        # 「很大的树」是四个连续汉字，但不在词表内，不能算固定四字表达
        self.assertEqual(sl.count_idioms("很大的树"), 0)
        m = sl.metrics("很大的树")
        self.assertEqual(m["idiom_pk"], 0.0)
        self.assertGreater(m["cluster4_pk"], 0)   # 只体现在粗代理里

    def test_cluster_metric_is_labelled_separately(self):
        m = sl.metrics("这件事一直悬而未决，拖了很久。")
        self.assertIn("cluster4_pk", m)
        self.assertIn("idiom_pk", m)
        self.assertNotEqual(m["cluster4_pk"], m["idiom_pk"])


class TestPerson(unittest.TestCase):
    """2. 第一人称口径"""

    def test_wo_self_not_double_counted(self):
        wo, self_, both, uniq = sl.person_counts("我自己知道")
        self.assertEqual((wo, self_, both), (1, 1, 1))
        self.assertEqual(uniq, 1)

    def test_wo_men_self_not_double_counted(self):
        wo, self_, both, uniq = sl.person_counts("我们自己决定")
        self.assertEqual(uniq, 1)

    def test_plain_self_counts_once(self):
        wo, self_, both, uniq = sl.person_counts("管好你自己")
        self.assertEqual((wo, self_, uniq), (0, 1, 1))

    def test_separate_metrics_exposed(self):
        m = sl.metrics("我自己想了很久，自己也想不明白。")
        self.assertAlmostEqual(m["wo_pk"], m["wo_pk"])       # 存在
        self.assertGreater(m["self_pk"], 0)
        self.assertIn("first_uniq_pk", m)
        # 去重值必须小于「我」+「自己」的粗暴相加
        naive = m["wo_pk"] + m["self_pk"]
        self.assertLessEqual(m["first_uniq_pk"], naive)


class TestSimile(unittest.TestCase):
    """3. 比喻误报"""

    def test_general_person_not_simile(self):
        strong, amb = sl.count_simile("一般人不会这么想。")
        self.assertEqual(strong, 0)

    def test_general_situation_not_simile(self):
        strong, amb = sl.count_simile("一般情况下他不会来。")
        self.assertEqual(strong, 0)

    def test_simile_marker_detected(self):
        strong, amb = sl.count_simile("他好似在等什么人。")
        self.assertEqual(strong, 1)

    def test_ambiguous_kept_out_of_simile(self):
        strong, amb = sl.count_simile("他好像在等什么人。")
        self.assertEqual(strong, 0)
        self.assertGreaterEqual(amb, 1)


class TestMarkdown(unittest.TestCase):
    """4. Markdown 标题不能被提前清理"""

    def test_heading_counted(self):
        text = "# 标题\n正文第一句。"
        self.assertEqual(sl.md_struct_lines(text), 1)

    def test_heading_reaches_verdict(self):
        res = sl.evaluate("# 标题\n正文第一句。", mode="note")
        self.assertTrue(any("Markdown" in h for h in res["hard"]))
        self.assertEqual(res["code"], 1)

    def test_bullet_counted(self):
        self.assertEqual(sl.md_struct_lines("- 第一条\n- 第二条"), 2)

    def test_corpus_note_line_removed_but_not_heading(self):
        text = "【说明】这是语料说明行\n正文一句。"
        self.assertNotIn("【说明】", sl.strip_corpus_notes(text))
        # 但 # 行必须留下
        self.assertIn("#", sl.strip_corpus_notes("# 标题\n正文"))


class TestSoftPhrases(unittest.TestCase):
    """5. 「首先/其次」只提醒，不硬性失败"""

    def test_first_second_is_warning_not_hard(self):
        text = "首先，我想说这件事。其次，还有别的。"
        res = sl.evaluate(text, mode="note")
        self.assertNotEqual(res["code"], 1)
        self.assertTrue(any("提醒级" in w for w in res["warns"]))

    def test_hard_phrase_still_fails(self):
        res = sl.evaluate("我们要赋能业务，形成闭环。", mode="note")
        self.assertEqual(res["code"], 1)


class TestModeSelection(unittest.TestCase):
    """6. 不同长度 / 体裁用不同标准"""

    def test_short_text_uses_note(self):
        self.assertEqual(sl.evaluate("今天去了学校。", mode="auto")["mode"], "note")

    def test_long_text_uses_long_essay(self):
        text = "这是一段用来测试长度判定的句子。" * 60
        self.assertEqual(sl.evaluate(text, mode="auto")["mode"], "long_essay")

    def test_explicit_mode_wins(self):
        self.assertEqual(sl.evaluate("今天去了学校。", mode="long_essay")["mode"], "long_essay")

    def test_baselines_differ(self):
        self.assertNotEqual(sl.BASELINE["long_essay"], sl.BASELINE["note"])

    def test_tolerance_scales_with_length(self):
        self.assertGreater(sl.tolerance(300), sl.tolerance(1200))
        self.assertGreater(sl.tolerance(1200), sl.tolerance(3000))


class TestExitCodes(unittest.TestCase):
    """7. 退出码与文档一致：0 通过 / 1 硬伤 / 2 偏离"""

    def test_code_pass(self):
        text = ("今天去了学校，路两边的树都黄了。风一吹，叶子就落下来。"
                "这样的天气，人容易想起以前之事。我坐在那儿，心里没什么波澜。"
                "是好是坏？我也不知道。")
        res = sl.evaluate(text, mode="note")
        self.assertEqual(res["code"], 0, msg="偏离项=%s 硬伤=%s" % (res["warns"], res["hard"]))
        self.assertEqual(res["verdict"], "PASS")

    def test_code_hard_ellipsis(self):
        res = sl.evaluate("我在等……", mode="note")
        self.assertEqual(res["code"], 1)

    def test_code_hard_book_title(self):
        res = sl.evaluate("我读了《道德经》。", mode="note")
        self.assertEqual(res["code"], 1)

    def test_code_deviation_is_two(self):
        # 无硬伤、但指标明显偏离（全用长句、无短句、无问句）
        text = ("这是一段故意写得非常长而且没有任何短句停顿的文字用来触发篇幅指标偏离，"
                "同样这一句也保持着相当的长度并且不给出任何疑问语气，"
                "第三句仍然很长很平没有任何变化，第四句依旧如此继续堆叠下去。")
        res = sl.evaluate(text, mode="long_essay")
        self.assertFalse(res["hard"])
        self.assertEqual(res["code"], 2)

    def test_verdict_mapping(self):
        self.assertEqual(sl.evaluate("今天去了学校。", mode="note")["verdict"] in
                         ("PASS", "DEVIATION"), True)


@unittest.skipUnless((ROOT / "references" / "corpus").exists(),
                     "references/corpus 不存在（不随公开仓库分发）")
class TestCorpusRegression(unittest.TestCase):
    """8. 语料回归：本人原文不应被判为硬伤"""

    def test_corpus_files_have_no_hard_issue(self):
        files = sorted((ROOT / "references" / "corpus").glob("*.txt"))
        self.assertGreaterEqual(len(files), 1)
        for p in files:
            text = p.read_text(encoding="utf-8")
            res = sl.evaluate(text, mode="note")
            self.assertEqual(res["hard"], [], msg="%s 被误判硬伤：%s" % (p.name, res["hard"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

# -*- coding: utf-8 -*-
"""
二、词元（Token）的观察与分析 —— 第 5 题
========================================
【题目】下面的代码片段被用来测试分词器对各种"困难文本"的处理能力。
请列举至少 3 种这段测试文本中包含的"困难情况"（如大小写、表情符号、编程代码等），
并说明为什么它们对分词器是挑战。

【答案】共 6 种：
1. 大小写 —— "English and CAPITALIZATION"
   同一个单词的大小写变体在词表中是**不同**词元。每种变体都单独占位会撑大词表、
   割裂同一语义；统一转小写又会丢掉"大写 = 专有名词 / 强调"的线索（Apple vs apple）。
   分词器必须在"压缩词表"和"保留大小写信息"之间取舍。
2. 英语以外的语言 —— "🕊鸟"（汉字）
   主流 BPE 词表以英文语料为主，对中日韩等**表意文字**覆盖很差。一个汉字常常找不到
   对应的整字词元，只能回退成 UTF-8 字节片段，一个字被切成 2~4 个词元，序列被拉长、
   语义碎片化，模型得"拼字节"才能还原一个字。
3. 表情符号 emoji —— "🕊"
   emoji 在 UTF-8 下占 4 个字节，字节级分词器会把它拆成 4 个无语义片段；而且 emoji 表
   每年还在新增，训练语料里出现极少，模型几乎学不到稳定的语义。
4. 编程代码 —— "show_tokens False None elif == >= else:"
   ① 标识符常用下划线 / 驼峰粘连、没有空格分隔，边界难判断；
   ② 复合运算符（==、>=）是不可分割的语法单元，被切成 = + = 就破坏了语义；
   ③ 关键字（elif、None、False）语义关键但形态极短，切错代价大；
   ④ 代码的字符分布与自然语言差别极大，通用语料的统计规律不适用。
5. 空白字符与缩进 —— "Two tabs:\t\t" "Three tabs:\t\t\t"
   Python 靠**缩进**表达语法结构，"两个 tab"和"三个 tab"是完全不同的程序结构。
   但分词时连续空白常被归一化 / 合并，缩进层级信息就被抹掉；制表符与空格混用还会
   造成切分不一致。
6. 数字 —— "12.0*50=600"
   数字的切分策略直接决定模型的算术能力：随机按字节切会让模型难以对齐个位 / 十位；
   退化成一位一个词元（1、2、.、0）更利于按位运算，但序列会变长。另外小数点、乘号、
   等号既像数字又像符号，边界模糊。
"""

import os

from transformers import AutoTokenizer

# 本地 models/ 下已有权重就直接读，没有则回退官方仓库名自动下载
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "deepseek-coder-1.3b-instruct")
if not os.path.isdir(MODEL):
    MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

# 图 5-1 里的测试文本
TEST_TEXT = (
    "English and CAPITALIZATION\n"
    "🕊鸟\n"
    "show_tokens False None elif == >= else:\n"
    'Two tabs:"\t\t"  Three tabs:"\t\t\t"\n'
    "12.0*50=600"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("测试文本：")
print(TEST_TEXT)
print("\n" + "=" * 60)

# 整段分词
ids = tokenizer(TEST_TEXT).input_ids
print(f"整段共切出 {len(ids)} 个词元\n")
for tid in ids:
    print(f"  {tokenizer.convert_ids_to_tokens(tid)!r}")

# 逐类看看各种"困难情况"各被切成几个词元
print("\n" + "=" * 60)
print("逐类切分结果：")
for label, text in [
    ("大小写     ", "English and CAPITALIZATION"),
    ("汉字       ", "鸟"),
    ("emoji      ", "🕊"),
    ("编程代码   ", "show_tokens False None elif == >= else:"),
    ("制表符     ", "\t\t"),
    ("数字       ", "12.0*50=600"),
]:
    sub = tokenizer(text, add_special_tokens=False).input_ids
    toks = tokenizer.convert_ids_to_tokens(sub)
    print(f"  {label} {text!r}")
    print(f"             → {len(sub)} 个词元 {toks}")

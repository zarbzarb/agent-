# -*- coding: utf-8 -*-
"""
二、词元（Token）的观察与分析 —— 第 3 题
========================================
【题目】下面的代码逐个打印输入中的每个词元：
        for id in input_ids[0]:
            print(tokenizer.decode(id))
（1）为什么第一个词元是 <s>？
（2）apolog、izing、trag、ic 这类词元说明了分词器具有什么特点？这样设计有什么好处？

【答案】
（1）因为分词器在编码时会**自动**在序列最前面插入 BOS 词元 <s>：
    统一标识序列起点、与训练语料的格式保持一致、便于多段文本的拼接与切分。
（2）说明该分词器采用**子词（subword）分词**（BPE / SentencePiece），而不是按整词或
    按字符切分：罕见的整词会被拆成若干个更常见、更小的子词片段。这样设计的好处：
      ① 彻底消除 OOV —— 任何字符串都能由子词拼出来，不必退化成 [UNK]；
      ② 词表规模可控 —— 只需要几万个词元就能覆盖多语言，比词级（动辄上百万）小得多，
         又比字符级序列短得多；
      ③ 复用词根词缀 —— apolog- / -izing / -ic 被反复使用，形态相近的词
         （apologize / apologizing / apology）共享向量，彼此"借力"，
         对没见过的词形也能泛化；
      ④ 英文平均一个词只切 1~3 个子词，序列不会过长，注意力 O(n²) 的开销也可控。
"""

import os

from transformers import AutoTokenizer

# 本地 models/ 下已有权重就直接读，没有则回退官方仓库名自动下载
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "deepseek-coder-1.3b-instruct")
if not os.path.isdir(MODEL):
    MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

prompt = ("Write an email apologizing to Sarah for the tragic gardening mishap. "
          "Explain how it happened.### Response:")
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# 题目里的原始写法：逐个打印每个词元
print("--- for id in input_ids[0]: print(tokenizer.decode(id)) ---")
for id in input_ids[0]:
    print(tokenizer.decode(id))

# 顺便看看子词是怎么切的（apolog / izing / trag / ic 就是从这里来的）
print("\n--- 词元 与 词元 ID ---")
for i, tid in enumerate(input_ids[0].tolist()):
    print(f"{i:>2}  {tokenizer.convert_ids_to_tokens(tid)!r:<22} id={tid}")

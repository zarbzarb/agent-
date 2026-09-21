# -*- coding: utf-8 -*-
"""
八、动手实践（选做，加分题） —— 第 13 题【交付物】
==================================================
【题目】参考本作业第一、二题的代码，完成以下实践并提交运行截图：
  · 使用 AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 加载分词器；
    → 按要求替换为 DeepSeek：AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
  · 对句子 "The dog chased the squirrel because it" 进行分词；
  · 打印每个词元及其对应的词元 ID；
  · 回答：it 被切分成了几个词元？为什么代词 it 的预测需要依赖前文（注意力机制的作用）？

【答案】
（一）it 被切分成了几个词元？
     **1 个**。it 是高频英文单词，本身完整地存在于分词器词表中，所以只会被切出
     **唯一一个**词元（id 固定），不会被拆成 "i" + "t" 或其它子词。
     （对比：罕见词才会被拆开，越常见的词在词表里越"完整"。）

（二）为什么代词 it 的预测需要依赖前文（注意力机制的作用）？
     因为代词**本身没有独立语义** —— it 到底指谁完全由上下文决定。在本句
     "The dog chased the squirrel because it" 中，it 既可以指 the dog 也可以指
     the squirrel，只有读了前文才可能判断。这正是**自注意力机制**要解决的问题：

     1) 上下文相关表示：Transformer 里每个词元的向量不是固定的词表向量，而是它与本句
        所有其他词元交互之后的**加权聚合**。计算 it 时，它的 Q 与前面所有词元
        （the / dog / chased / the / squirrel / because）的 K 做点积得到注意力权重 α，
        再用 α 对它们的 V 加权求和 —— 于是 it 的表示里"混入"了 dog 和 squirrel 的信息，
        而不是孤立的"它"。
     2) 因果遮罩（causal mask）：自回归模型只允许看到自己及**之前**的词元，未来位置被
        mask 成 −∞。所以"依赖前文"是机制上的**硬约束**。
     3) 多层堆叠 → 逐步消歧：浅层注意力捕捉语法搭配（it 与 chased / because 的关系），
        深层才完成**指代消解**（判断 it 更可能是 squirrel，因为松鼠会爬树逃走、狗在追）；
        逐层精炼后，lm_head 最终才输出 climbed / ran / escaped 这类符合松鼠行为的词。
     4) 反证：把前文换成 "…because it was hungry."，模型会偏向 the dog；换成
        "…because it was scared."，则偏向 the squirrel。**同一个词、同一位置，
        只因上下文不同，内部表示与预测结果就不同** —— 这就是"必须依赖前文"最直观的证据。
"""

import os

from transformers import AutoTokenizer

# 本地 models/ 下已有权重就直接读，没有则回退官方仓库名自动下载
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "deepseek-coder-1.3b-instruct")
if not os.path.isdir(MODEL):
    MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
SENTENCE = "The dog chased the squirrel because it"
print(f"词表大小：{len(tokenizer):,}\n")

# ---- 对句子分词 ----
input_ids = tokenizer(SENTENCE, return_tensors="pt").input_ids
print(f"句子：{SENTENCE!r}")
print(f"input_ids = {input_ids}\n")

# ---- 打印每个词元及其对应的词元 ID ----
print(f"{'序号':<6}{'词元 (token)':<24}{'词元 ID':<12}")
print("-" * 44)
for i, tid in enumerate(input_ids[0].tolist()):
    print(f"{i:<6}{tokenizer.convert_ids_to_tokens(tid)!r:<24}{tid:<12}")

# ---- 回答问题 (一)：it 被切成几个词元 ----
it_ids = tokenizer("it", add_special_tokens=False).input_ids
print(f"\n【答案(一)】tokenizer('it', add_special_tokens=False).input_ids = {it_ids}")
print(f"          → it 被切分成 {len(it_ids)} 个词元：{tokenizer.convert_ids_to_tokens(it_ids)}")

# 对比：罕见词才会被拆成多个子词
print("\n对比（罕见词会被拆成多个子词）：")
for w in ["squirrel", "apologizing", "antidisestablishmentarianism"]:
    ids = tokenizer(w, add_special_tokens=False).input_ids
    print(f"  {w:<32} → {len(ids)} 个词元 {tokenizer.convert_ids_to_tokens(ids)}")

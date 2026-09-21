# -*- coding: utf-8 -*-
"""
二、词元（Token）的观察与分析 —— 第 4 题（填空题）
==================================================
【题目】分词器中包含若干特殊词元，请根据下表完成填空。

【答案】
  unk（未知）  ：未知词元 [UNK]。当某个字符/词语在词表里查不到、又无法拆成已知子词时
                用它占位，表示"不认识的内容"。
  sep（分隔符）：分隔符词元 [SEP]。用于需要把**两段文本**一起送给模型的特定任务
                （句对分类、问答、句子相似度重排序等，这类场景下的模型被称为
                 交叉编码器 cross-encoder），把两段文本分隔开。
  pad（填充）  ：填充词元 [PAD]。因为模型通常要求输入具有**固定长度**（上下文窗口），
                一个 batch 内的短句要补齐到最长句的长度，就用它填充**未使用的位置**，
                并配合 attention_mask 让模型忽略这些位置。
  cls（分类）  ：分类词元 [CLS]。主要用于**分类任务**的特殊词元，放在句首，
                其最后一层的向量被当作整句的聚合表示，接一个分类头即可输出类别。
  mask（掩码） ：掩码词元 [MASK]。在（预）训练过程中用于**隐藏词元**，让模型根据上下文
                预测被遮住的词（如 BERT 的 MLM 任务），从而学到双向语义表示；推理时不使用。

【补充】DeepSeek 使用 Llama 风格分词器，不用 BERT 这套 [CLS]/[SEP]/[MASK]，
        而是用 <s>、</s>、<｜begin▁of▁sentence｜>、<｜end▁of▁sentence｜>、<｜Assistant｜>
        这些控制标记；但"需要一个专门词元来承担某个功能"的设计思想是通用的。
"""

import os

from transformers import AutoTokenizer

# 本地 models/ 下已有权重就直接读，没有则回退官方仓库名自动下载
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "deepseek-coder-1.3b-instruct")
if not os.path.isdir(MODEL):
    MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print(f"本作业使用的分词器：{tokenizer.name_or_path}")
print(f"词表大小：{len(tokenizer):,}\n")

# 概念表（题目要求填空的内容）
print("【填空答案】")
for name, use in [
    ("unk（未知）",   "词表里查不到、又无法切成已知子词时用它占位"),
    ("sep（分隔符）", "把两段文本分隔开，供句对分类/问答/句子相似度等任务使用"),
    ("pad（填充）",   "把 batch 内不同长度的句子补齐到同一长度，占位用，模型会忽略它"),
    ("cls（分类）",   "放在句首，其最终向量作为整句的聚合表示，供分类任务使用"),
    ("mask（掩码）",  "预训练时遮住要预测的词，让模型学会从上下文预测（MLM）"),
]:
    print(f"  {name:<16} {use}")

# 本作业的 DeepSeek（Llama 风格）分词器实际有哪些特殊词元
print("\n【DeepSeek 分词器实际的特殊词元】")
for tid in tokenizer.all_special_ids:
    tok = tokenizer.convert_ids_to_tokens(tid)
    if tok is not None:
        print(f"  id={tid:<7} {tok!r}")

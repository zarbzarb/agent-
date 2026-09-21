# -*- coding: utf-8 -*-
"""
三、词元嵌入 —— 第 6 题
=======================
【题目】使用 DeBERTa 模型处理句子 "Hello world"，得到输出后执行 output.shape，
        结果为 torch.Size([1, 4, 384])。
（1）请解释这 3 个维度分别是什么含义。
（2）句子只有两个单词，为什么词元数是 4？

【答案】
（1）| 维度   | 数值 | 含义 |
        | 第 1 维 | 1   | batch size（批次大小），本次只送了 1 个句子；设这一维是为了能一次并行处理多句 |
        | 第 2 维 | 4   | sequence length（序列长度 / 词元数），这句话共 4 个词元。注意是"词元数"而不是"单词数" |
        | 第 3 维 | 384 | hidden size / embedding size（隐藏维度），每个词元被表示成一个 384 维稠密向量
        （DeBERTa-v3-xsmall 的宽度就是 384）
    一句话：1 个批次 × 4 个词元 × 每个词元 384 维向量。
（2）因为 DeBERTa 的分词器会自动在句子首尾加上两个特殊词元：
        [CLS]  Hello  world  [SEP]
    · [CLS] 放在最前面，供分类任务使用，其最终向量被当作整句的聚合表示；
    · [SEP] 放在最后面，是句子结束 / 分隔标记（输入两段文本时也用它隔开）。
    所以 2 个实词 + 2 个特殊词元 = 4 个词元，输出的序列长度是 4 而不是 2。
"""

import os

import torch
from transformers import AutoModel, AutoTokenizer

# 本地 models/ 下已有权重就直接读，没有则回退官方仓库名自动下载
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "deberta-v3-xsmall")
if not os.path.isdir(MODEL):
    MODEL = "microsoft/deberta-v3-xsmall"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)   # 用原生 slow 分词器，切分结果一致
model = AutoModel.from_pretrained(MODEL)
model.eval()

inputs = tokenizer("Hello world", return_tensors="pt")
print("分词结果 tokens =", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

with torch.no_grad():
    output = model(**inputs)

print("\noutput.last_hidden_state.shape =", output.last_hidden_state.shape)
print("  = [batch_size, seq_len, hidden_size] = [1, 4, 384]")
print("\n注意：句子只有 2 个单词，但分词器自动加了 [CLS] 与 [SEP]，所以词元数是 4。")

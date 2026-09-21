# -*- coding: utf-8 -*-
"""
二、词元（Token）的观察与分析 —— 第 2 题
========================================
【题目】打印 input_ids 看看它包含什么。
（1）input_ids 的类型是什么？device='cuda:0' 说明它被放在了哪里？
（2）张量中的每个数字（如 14350、385）代表什么含义？

【答案】
（1）类型是 torch.Tensor（dtype=torch.int64 的二维张量），形状 [1, 25]：
        第 1 维 1  = batch size（批次大小），本次只送了 1 个句子；
        第 2 维 25 = sequence length（序列长度），共 25 个词元。
    device='cuda:0' 说明这个张量被放在**第 0 号 GPU 的显存**里。
    PyTorch 要求张量与模型处在同一设备上才能运算，所以要 .to("cuda")；没有 GPU 时是 cpu。
（2）每个数字是**词元 ID（token id）**，即该词元在分词器词表中的下标。
    分词器内部维护一张"字符串 ↔ 整数"的映射表（"Write"→14350、"an"→385 …）。
    模型不认识文字、只认识这些整数：输入时靠整数去嵌入表里取向量，输出时再 decode 回文字。
"""

import os

import torch
from transformers import AutoTokenizer

# 本地 models/ 下已有权重就直接读，没有则回退官方仓库名自动下载
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "deepseek-coder-1.3b-instruct")
if not os.path.isdir(MODEL):
    MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

# 有 GPU 用 GPU，没有则自动退回 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

prompt = ("Write an email apologizing to Sarah for the tragic gardening mishap. "
          "Explain how it happened.### Response:")
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

print("input_ids =", input_ids)
print()
print("类型   type   =", type(input_ids))
print("数据类型 dtype =", input_ids.dtype)
print("形状   shape  =", tuple(input_ids.shape), " → [batch_size, seq_len]")
print("设备   device =", input_ids.device)

# 看看这些数字分别对应什么文字，验证"数字 = 词表中的下标"
print("\n前 10 个数字对应的文字：")
for tid in input_ids[0].tolist()[:10]:
    print(f"  {tid:<7}{tokenizer.decode([tid])!r}")

# -*- coding: utf-8 -*-
"""
一、文本生成与分词流程 —— 第 1 题
==================================
【题目】
（1）补全下面的代码：对 prompt 进行分词得到 input_ids，再用模型生成 20 个新词元，
    最后解码并打印输出。
        input_ids = ______________________________
        generation_output = model.generate(input_ids=input_ids, ______________________)
        print(tokenizer.decode(______________))
（2）运行结果输出开头的 <s> 是什么？它是否需要我们手动输入？

【答案】
（1）补全见下面标有「补全」的三行：
    ① 分词：input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    ② 生成：max_new_tokens=20
    ③ 解码：tokenizer.decode(generation_output[0])
（2）<s> 是 BOS（Beginning Of Sequence，"序列起始"）特殊词元。它的作用是标识
    "一个新序列从这里开始"，并保证推理时的输入格式与预训练时保持一致。
    **不需要手动输入** —— 分词器在编码时会自动把它插到序列最前面。
    （DeepSeek 同为 Llama 风格分词器，把 BOS 写作 <｜begin▁of▁sentence｜>，
      名称不同、作用完全相同。）

【模型替换】原作业为 microsoft/Phi-3-mini-4k-instruct，按要求换成国产 DeepSeek：
    deepseek-ai/deepseek-coder-1.3b-instruct

【说明】1.3B 的代码专用模型面对"写邮件"这类写作任务会委婉拒绝（输出 "I'm sorry…"），
        这是小模型容量与训练目标的限制，**不影响本题要观察的分词流程与 BOS 机制**。
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型目录：本地 models/ 下已有权重就直接读（离线可用）；
# 别人克隆后没有权重时，回退到官方仓库名自动联网下载（国内可先 set HF_ENDPOINT=https://hf-mirror.com）
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "deepseek-coder-1.3b-instruct")
if not os.path.isdir(MODEL):
    MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

# 有 GPU 就用 GPU，没有则自动退回 CPU（device_map 与张量都跟着它走）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型及其分词器
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map=DEVICE,
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# 提示词（原作业末尾的 <|assistant|> 是 Phi-3 的回复起始标记，
# DeepSeek 分词器对应的回复起始标记是 "### Response:"）
prompt = ("Write an email apologizing to Sarah for the tragic gardening mishap. "
          "Explain how it happened.### Response:")

# ---------- (1) 补全的代码 ----------
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)     # 补全① 分词

generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=20,                                                      # 补全② 生成 20 个新词元
)

print(tokenizer.decode(generation_output[0]))                              # 补全③ 解码

# ---------- (2) 观察 BOS ----------
print("\n序列第 1 个词元 =", repr(tokenizer.decode(input_ids[0, 0])),
      "← BOS，由分词器自动添加，不需要手动输入")
print("序列共", input_ids.shape[1], "个词元，其中新生成 20 个")

# -*- coding: utf-8 -*-
"""
七、查看 LLM 内部：模型结构与 KV 缓存 —— 第 11 题
=================================================
【题目】下面是打印 Phi-3 模型结构得到的输出，请据此回答（本作业已把 Phi-3 换成 DeepSeek）：
（1）词元嵌入矩阵 embed_tokens 的形状为 Embedding(32064, 3072)，两个数字分别代表什么？
（2）模型堆叠了多少个 Phi3DecoderLayer（Transformer 块）？
（3）lm_head 的输出维度是 32064，为什么恰好等于词表大小？它的输出有什么用途？

【答案】
（1）32064 = **词表大小 vocab_size**，模型一共认识 32064 个词元，**每个词元占一行**；
    3072  = **嵌入维度 hidden size（d_model）**，每个词元被映射成的稠密向量的长度。
    也就是说它是一张 32064 行 × 3072 列的查找表：输入 token id（0~32063 的整数），
    取出该 id 对应那一行的 3072 维向量，作为后续所有层的输入表示。
（2）**32 层**。证据是 `(layers): ModuleList((0-31): 32 x Phi3DecoderLayer(...))` —— 下标 0~31。
    每层包含两个子层：注意力子层（qkv_proj 把 3072 维一次映射成 3×3072，经注意力后由
    o_proj 映射回 3072 维）与前馈子层（SwiGLU：gate_up_proj 把 3072 扩到 2×8192，
    门控相乘后降到 8192，再由 down_proj 回到 3072）。每个子层前后有 RMSNorm 与残差连接。
（3）因为语言模型的最终任务是**预测下一个词元**：
    ① 经过 32 层 Transformer 与最后的 norm，序列的每个位置都有一个 3072 维向量；
    ② lm_head 是一个 3072 → 32064 的线性层，把它映射成 **32064 个分数（logits）**，
       第 i 个分数表示"下一个词元是词表里第 i 个 token"的倾向；
    ③ 经 softmax 得到**整个词表上的概率分布**，再按贪心或采样挑出一个 token；
    ④ 把新 token 接到输入末尾喂回模型，循环往复，就得到整段**自回归生成**。
    所以输出维度**必须等于词表大小** —— 词表里有多少候选 token，就至少要输出多少个分数。
    另外会发现 lm_head 与 embed_tokens 的形状互为转置，很多模型因此**共享权重**
    （weight tying），既省显存又能提升小模型的效果。

【本作业实测（DeepSeek）】deepseek-ai/deepseek-coder-1.3b-instruct 是 LlamaForCausalLM 结构：
    embed_tokens = Embedding(32256, 2048)、24 层 LlamaDecoderLayer、
    lm_head = Linear(2048, 32256)。三问的形式与答案完全一致：行数 = 词表大小，
    列数 = 嵌入维度，lm_head 的输出维度 = 词表大小。
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 本地 models/ 下已有权重就直接读，没有则回退官方仓库名自动下载
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "deepseek-coder-1.3b-instruct")
if not os.path.isdir(MODEL):
    MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"

# 有 GPU 用 GPU，没有则自动退回 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型及其分词器
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map=DEVICE,
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# ---- 题目要求：打印模型结构 ----
print(model)

# ---- 关键信息 ----
embed = model.get_input_embeddings()
n_layers = len(model.model.layers)

print("\n" + "=" * 60)
print(f"词表大小 (vocab_size) = {len(tokenizer):,}"
      f"（模型嵌入表按 64 取整到 {embed.num_embeddings:,}，故略大于分词器词表）")
print(f"embed_tokens          = {embed}")
print(f"DecoderLayer 层数     = {n_layers}")
print(f"lm_head               = {model.lm_head}")

# ---- 实测：张量形状是怎么流动的 ----
ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(DEVICE)

with torch.no_grad():
    hidden = model.model.embed_tokens(ids)
    logits = model.lm_head(hidden)

print(f"\n输入 input_ids       : {tuple(ids.shape)}")
print(f"过 embed_tokens 之后 : {tuple(hidden.shape)}   ← 每个词元变成 {hidden.shape[-1]} 维向量")
print(f"过 lm_head 之后      : {tuple(logits.shape)}   ← 每个位置给出整个词表的分数")

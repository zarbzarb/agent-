# -*- coding: utf-8 -*-
"""
七、查看 LLM 内部：模型结构与 KV 缓存 —— 第 12 题
=================================================
【题目】下面的代码用 %%timeit 测量生成 100 个词元所需的时间（原作业 Phi-3 实测：
        启用缓存约 4.5 秒，禁用缓存约 21.8 秒）。
（1）请解释键值（KV）缓存为什么能大幅加速生成。
（2）为什么在生成第二个词元时，模型只需要计算新词元的缓存，而不必重新计算之前的所有词元？

【答案】
（1）自回归生成是**逐词元**进行的，每吐出一个新词元就要再跑一次前向。而自注意力中
    每个位置都要做：
        Attention(Q, K, V) = softmax(Q Kᵀ / √d) V
    即"用当前词元的 Q 去和**之前所有词元**的 K 做点积得到权重，再用这些权重对**之前所有
    词元**的 V 加权求和"。
    · **不使用缓存**：为了得到第 t 步的 K、V，必须把前面 t−1 个词元的 K、V **重新算一遍**，
      第 t 步的计算量正比于 t；生成 n 个词元的总量 ≈ 1+2+…+n = **O(n²)**，绝大部分是重复劳动。
    · **使用缓存**：注意**第 i 个词元的 K、V 只依赖第 1..i 个词元**，与后面新生成的词元无关
      （因果注意力看不到未来），一旦算出来就是**常量**，可以存起来反复复用。于是每步只需为
      新增的那**一个**词元计算 K、V 并**追加**到缓存，再让它与缓存中全部 K、V 做注意力。
      单步从 O(t) 降到 **O(1)**，整体从 **O(n²) 降到 O(n)**。
    这就是 100 个词元能大幅加速的原因。作业参考值是 21.8 秒 → 4.5 秒（约 4.8×）。
    代价：KV 缓存占用的显存 ≈ 2 × 层数 × 序列长度 × 隐藏维度 × 字节数，随序列长度**线性增长**，
    所以长上下文很吃显存，工程上才有了 MQA / GQA、PagedAttention、缓存量化等优化。
（2）因为**因果（causal）注意力**保证"过去不受未来影响"：第 i 个位置的 K、V 在每一层都
    只由第 1..i 个词元决定，一旦算完就固定不变。具体到这一步：
    · 第一步处理 prompt 时（prefill），模型已经把 prompt 里**所有**词元的 K、V 在每一层
      算好并存入缓存；
    · 生成第二个词元时，真正"新"的只有上一步刚生成的那一个词元，它需要（在每一层）计算
      自己的 Q、K、V；而 prompt 部分的 K、V **直接从缓存读取**，无需重算；
    · 于是这一步的计算量只与"新增的 1 个词元"有关，与 prompt 有多长**无关**；
      prompt 越长，省下的重复计算越多，加速越明显。
    公式角度：o_t = Σ_{i≤t} softmax(q_t · k_iᵀ / √d) · v_i，其中 {k_i, v_i}_{i<t}
    全部来自缓存，本步只新算 q_t, k_t, v_t。
    这也是"缓存"名字的由来：缓存的是每层的 **K**ey 和 **V**alue；Q 不需要缓存，
    因为每个位置的 Q 只用一次，用完即丢。
"""

import os
import time

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

MAX_NEW_TOKENS = 100

# 对输入提示词进行分词（图 12-3 的代码）
prompt = ("Write a very long email apologizing to Sarah for the tragic gardening mishap. "
          "Explain how it happened.")
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
print(f"prompt 词元数 = {input_ids.shape[1]}，要生成 {MAX_NEW_TOKENS} 个词元\n")


def sync():
    """GPU 上计时前必须同步，否则测到的是"下发命令"的时间；CPU 上不需要。"""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def bench(use_cache):
    """测量生成 100 个词元所需的时间（对应作业里的 %%timeit）。"""
    sync()
    t0 = time.time()
    with torch.no_grad():
        model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=use_cache,
        )
    sync()
    return time.time() - t0


t_with = bench(use_cache=True)
t_without = bench(use_cache=False)

print(f"use_cache=True  （启用 KV 缓存）：{t_with:.2f} 秒")
print(f"use_cache=False （禁用 KV 缓存）：{t_without:.2f} 秒")
print(f"\n加速比 = {t_without / t_with:.2f}×")
print("（原作业 Phi-3 的参考值：4.5 秒 vs 21.8 秒，约 4.8×；模型、显卡、prompt 长度不同，"
      "数值会有差异，但 O(n²) → O(n) 的量级差异一致。）")

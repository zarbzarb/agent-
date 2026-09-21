# 《NLP 词元和嵌入》作业 · DeepSeek 版

8 大题 13 小题，**每题一个可直接运行的 Python 源文件**，只做题目要求的事。
作业中原本"调用大模型"的环节（Phi-3）已按要求全部替换为国产大模型 **DeepSeek（深度求索）**。

- **题目答案** → [`ANSWERS.md`](ANSWERS.md)
- **题目源码** → `q01_*.py` ~ `q13_*.py`（每题文件开头 docstring 里也写了本题答案）
- **第 13 题交付物** → [`q13_practice.py`](q13_practice.py)

> 没有公共配置模块：每道题都按作业原文的写法，在自己文件里直接
> `AutoModelForCausalLM.from_pretrained(...)` / `AutoTokenizer.from_pretrained(...)`。

---

## 1. 怎么跑

```bash
git clone https://github.com/zarbzarb/agent-.git
cd agent-/deepseek_nlp_homework
pip install -r requirements.txt
python q13_practice.py          # 第 13 题，只需分词器，几秒就能出结果
```

**模型权重不用手动准备**：脚本会先看脚本同级的 `models/` 目录下有没有权重；
没有就自动回退到官方仓库名联网下载（首次约 2.6GB，国内建议先设镜像）：

```bash
set HF_ENDPOINT=https://hf-mirror.com        # Windows
export HF_ENDPOINT=https://hf-mirror.com     # macOS / Linux
```

**没有 GPU 也能跑**：`device_map` 会自动退回 CPU，张量设备跟着一起切。
（只有第 12 题在 CPU 上会非常慢，因为要对比"生成 100 个词元"的耗时，属正常现象。）

13 题逐题运行：

| 题号 | 命令 | 备注 |
|---|---|---|
| 1 | `python q01_text_generation.py` | 加载 DeepSeek 模型，约 2.6GB |
| 2 | `python q02_input_ids.py` | 只需分词器 |
| 3 | `python q03_tokens.py` | 只需分词器 |
| 4 | `python q04_special_tokens.py` | 只需分词器 |
| 5 | `python q05_hard_text.py` | 只需分词器 |
| 6 | `python q06_token_embedding.py` | 首次会下载 DeBERTa-v3-xsmall（230MB） |
| 7 | `python q07_sentence_embedding.py` | 首次会下载 all-mpnet-base-v2（418MB） |
| 8 | `python q08_glove.py` | 首次会下载 GloVe（66MB） |
| 9 | `python q09_word2vec_train.py` | 训练约 3~4 分钟 |
| 10 | `python q10_song_recommend.py` | **需先跑第 9 题**（依赖它产出的 `song2vec.model`） |
| 11 | `python q11_model_structure.py` | 加载 DeepSeek 模型 |
| 12 | `python q12_kv_cache.py` | 加载 DeepSeek 模型，对比 KV 缓存耗时 |
| 13 | `python q13_practice.py` | ★ 交付物 |

各题完整运行结果存在 `outputs/` 下，可作为"运行截图"的依据。

### 从零装环境

```bash
conda create -n py3_11 python=3.11 -y
conda activate py3_11
pip install torch --index-url https://download.pytorch.org/whl/cpu      # 无 GPU 时用这行，体积小
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 2. DeepSeek 替换说明

| 原作业（Phi-3） | 本作业（DeepSeek） |
|---|---|
| `microsoft/Phi-3-mini-4k-instruct` | `deepseek-ai/deepseek-coder-1.3b-instruct`（1.3B / 2.6GB，6GB 显存或 CPU 都能跑） |
| 加载方式 `AutoModelForCausalLM.from_pretrained(..., device_map="cuda", torch_dtype="auto", trust_remote_code=True)` | **完全一样**，只换模型名字符串 |
| `<s>`（BOS 起始词元） | `<｜begin▁of▁sentence｜>`（同样由分词器**自动添加**，作用完全一致） |
| `<|assistant|>` 回复标记 | `### Response:`（DeepSeek 分词器的回复起始标记） |
| `model.generate(..., use_cache=True/False)` | **完全一样** |

如果本机已经离线下好权重，直接放在脚本同级 `models/` 下即可（`deepseek-coder-1.3b-instruct`、
`deberta-v3-xsmall`、`all-mpnet-base-v2`），脚本会优先读本地，不再联网。

**换成分词器后答案为什么还成立？** DeepSeek 用的是 **Llama 风格 BPE 分词器**，与 Phi-3 同族。
实测的切分结果：

```
apolog + izing     trag + ic     gard + ening     m + ish + ap     Expl + ain
```

与作业截图里 Phi-3 的切分**几乎一模一样**，所以第 1~5 题关于特殊词元、子词切分的答案框架完全成立。

---

## 3. 文件清单

| 文件 | 题号 | 内容 |
|---|---|---|
| `q01_text_generation.py` | 第 1 题 | 分词 → 生成 20 个新词元 → 解码；BOS 说明 |
| `q02_input_ids.py` | 第 2 题 | input_ids 的类型 / device / 逐个数字含义 |
| `q03_tokens.py` | 第 3 题 | 逐个打印词元 + 子词切分 |
| `q04_special_tokens.py` | 第 4 题 | 5 个特殊词元用途（填空）+ DeepSeek 实际特殊词元 |
| `q05_hard_text.py` | 第 5 题 | 6 类"困难文本"切分实验 |
| `q06_token_embedding.py` | 第 6 题 | DeBERTa 输出 `[1, 4, 384]` 的含义 |
| `q07_sentence_embedding.py` | 第 7 题 | SentenceTransformer 输出 `(768,)` 的含义 |
| `q08_glove.py` | 第 8 题 | GloVe `most_similar("king")` + 向量类比 |
| `q09_word2vec_train.py` | 第 9 题 | 加载歌单数据 + 训练 Word2vec |
| `q10_song_recommend.py` | 第 10 题 | `print_recommendations(3822)` + 为什么合理 |
| `q11_model_structure.py` | 第 11 题 | 打印模型结构，回答 embed / 层数 / lm_head |
| `q12_kv_cache.py` | 第 12 题 | KV 缓存计时对比（use_cache True/False） |
| `q13_practice.py` | 第 13 题 | ★ 分词 + 打印词元 ID + 回答两问 |
| `outputs/q01~q13.txt` | — | 各题完整运行记录 |
| `ANSWERS.md` | — | 13 题全部答案（填空 + 简答） |
| `requirements.txt` | — | 依赖清单 |

---

## 4. 常见问题

| 现象 | 解决 |
|---|---|
| 首次运行卡在"下载模型" | 正常，DeepSeek 权重 2.6GB；国内先 `set HF_ENDPOINT=https://hf-mirror.com` |
| 显存不足（CUDA out of memory） | 用 1.3B 模型；脚本会自动退回 CPU，也可手动改 `DEVICE = "cpu"` |
| 第 8 题下载 GloVe 慢 | 脚本优先读 `data/` 下已下载好的文件，没有才走 `api.load()` 联网下载 |
| 第 9/10 题数据集下载失败 | 脚本会自动从作业给出的地址下载到 `data/`；可手动下载后放进去 |
| 第 10 题报错找不到 `song2vec.model` | 先跑第 9 题 |
| 第 1 题模型输出"拒绝写邮件" | 1.3B 代码专用模型对写作任务偶尔会委婉拒绝，是模型容量的正常表现，**不影响本题要观察的分词流程与 BOS 机制** |
| 第 12 题跑很久 | 禁用 KV 缓存那一次本来就是 O(n²)，GPU 上也要几十秒，CPU 上更慢，属正常 |

---

## 5. 提交说明

1. **填空题与简答题**：参考 [`ANSWERS.md`](ANSWERS.md)。
2. **第 13 题**（要求提交 .py + 运行截图）：提交 `q13_practice.py` 与运行截图。
3. **不提交这些大文件**（已在 `.gitignore` 里排除，运行时自动下载）：
   - `models/`（约 3.2GB 权重）
   - `data/`（GloVe 163MB + 歌单数据集 13MB）
   - `song2vec.model`（第 9 题训练产物，20MB）

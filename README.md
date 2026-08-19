# 景点讲解 AI（直连式）

> 面向游客的实时景点讲解系统：流式文本 + 语音播报，RAG 检索增强问答。中文原生、低延迟。

本项目采用**直连式架构**：`web/server.py` 直接调用智谱 embedding-3 做向量化、DeepSeek 做流式问答、本地余弦检索，零 Docker / Ollama / AnythingLLM 依赖，仅需 Python 标准库即可运行。方案细节见 `docs/最优配置方案.md`（其中 AnythingLLM 相关章节为遗留方案，仅作设计参考）。

## 功能特性

- 🎯 **景点知识库**：内置广州景点（当前 11 个在册，sample-data 另有已删除景点的文档可恢复）；文件夹式管理（新建景点 / 上传文档 / 删除景点）
- ✂️ **每景点切分配置**：设置 → 通用 → 每个景点「⚙ 切分设置」可独立配置文档切分参数（最大段字符数 / 重叠字符数）；长文档自动切多段分别向量化，保存后立即重嵌，提高检索召回
- 💬 **流式讲解**：DeepSeek 流式输出，首字即回传，引用来源
- 🔊 **语音播报**：整段朗读不丢字、新回答打断旧播报；语速 0.5–3.0、音色可选、可试听
- 📄 **文档上传**：md / docx / pdf（20MB 内）上传即解析入知识库——docx/pdf 用纯标准库提取文本后自动分块嵌入；图片记录文件名与归属（不识别内容）；扫描件 PDF 无法提取文本会明确提示
- 🔐 **密钥加密存储**：PBKDF2 + HMAC 认证加密，落盘非明文
- 🎨 **界面**：三栏布局、深色模式、字号调节、收藏、搜索、快捷提问、演示模式兜底
- ⚡ **缓存**：向量磁盘缓存（增量嵌入）+ LRU 语义响应缓存，重复问题免调用 LLM
- 📊 **效果评测**：内置 29 题评测集（检索命中 + 端到端问答），一键跑 3 轮出报告，见 `eval/REPORT.md`

## 快速开始

```bash
# 1. 申请两个 Key：
#    DEEPSEEK_API_KEY   https://platform.deepseek.com
#    ZHIPU_API_KEY      https://open.bigmodel.cn   (智谱 embedding-3)
cp .env.example .env   # 填入两个 Key

# 2. 启动（二选一）
python web/server.py        # 命令行启动
# 或双击 start.bat          # Windows 双击启动：pythonw 后台运行，日志写入 server_run.log，
                            # 关闭弹出的黑窗口不会停止服务；停止服务请结束 pythonw.exe

# 3. 浏览器打开 http://127.0.0.1:8080
```

> 不填 Key 也能打开网页浏览界面（自动进入演示模式），但真实讲解需配置 Key（环境变量或网页「设置」里录入）。

> ⚠️ **安全提示**：服务默认仅监听 `127.0.0.1`（本机）。若需局域网多人访问，用 `python web/server.py --host 0.0.0.0`，但**所有接口无鉴权、API Key 以明文 HTTP 传输**，请勿在公网或不受信任的网络暴露，也不要在服务器上存放与本项目无关的敏感文件。

## 目录结构

```
├── web/
│   ├── server.py              # 直连后端（纯标准库：嵌入/检索/流式/上传/加密/缓存）
│   ├── extract.py             # 纯标准库文本提取（docx 解 zip+xml / pdf 文本流启发式）
│   └── index.html             # 前端（三栏布局/流式/语音/文档管理/设置抽屉）
├── scripts/
│   ├── preprocess.py          # 景点文档清洗 + 结构化分块
│   ├── run_e2e.py             # 直连端到端验证（零依赖）
│   └── evaluate_rag.py        # RAG 评测：检索命中 + 端到端问答，多轮可复跑
├── eval/
│   ├── questions.json         # 评测集（29 题：11 景点 × 门票/开放/交通/看点/历史 + 库外负样本）
│   ├── REPORT.md              # 评测报告（3 轮汇总 + 逐题明细 + 失败案例）
│   └── results/               # 每轮 CSV（round1/2/3.csv，可人工复核）
├── examples/
│   ├── demo_chat.py           # AnythingLLM 流式文本客户端（遗留，需 Docker）
│   └── demo_voice.py          # AnythingLLM 流式语音客户端 Edge-TTS（遗留，需 Docker）
├── sample-data/               # 内置 12 个景点知识文档（## 景点名 + 固定字段）
├── deploy/docker-compose.yml  # AnythingLLM 部署（遗留/未维护替代方案）
├── docs/
│   ├── 最优配置方案.md          # 完整方案（选型/分块/流式/缓存/预处理/参数表；含遗留的 AnythingLLM 章节）
│   └── 界面设计方案.md          # 界面设计（布局/令牌/交互/断点/验收清单）
├── tests/                     # 单元测试（61 项）
├── .env.example               # 环境变量示例
├── ruff.toml                  # 质检配置
└── start.bat                  # Windows 双击启动脚本
```

## 命令行工具

```bash
# 直连端到端验证（零依赖，读取 DEEPSEEK_API_KEY / ZHIPU_API_KEY）
python scripts/run_e2e.py "广州塔的门票多少钱？"

# 文档预处理：清洗 + 按景点条目分块（500~800 字符 / 50 重叠）
python scripts/preprocess.py 原始文档目录 -o sample-data
```

AnythingLLM 客户端（**遗留**，需先按 §AnythingLLM 部署启动 Docker）：

```bash
pip install requests edge-tts
export ALLM_BASE=http://127.0.0.1:3001
export ALLM_KEY="AnythingLLM 设置页生成的 API Key"

python examples/demo_chat.py "介绍一下陈家祠的历史"   # 流式文本
python examples/demo_voice.py "给我讲讲广州塔"        # 流式语音(生成 讲解音频.mp3)
```

## 核心选型（速记）

| 组件 | 选型 | 理由 |
|---|---|---|
| 嵌入模型 | **智谱 embedding-3**（云端） | 中文原生，2048 维，无需本地 Ollama |
| 聊天模型 | **deepseek-chat**（云端） | 中文强 + 流式快 + 极低成本 |
| 检索 | **本地余弦检索 Top-4 + 名称加权** | 景点库小，无需独立向量服务；查询含景点名时相似度 +0.15 修正排序 |
| 向量缓存 | **磁盘 pickle + 增量嵌入** | 新增文档才重新向量化，重启秒加载 |
| 分块 | 500~800 字符 / 50 重叠 | 按景点条目语义切分 |
| 流式 | DeepSeek stream + 整段语音 | 首字即回传，播报不丢字 |
| 缓存 | LRU 语义响应缓存 | 高频问题（门票/开放时间）免调用 LLM |

## 环境变量

| 变量 | 说明 | 必填 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek 聊天模型 Key | 直连 ✅ |
| `ZHIPU_API_KEY` | 智谱 embedding-3 Key | 直连 ✅ |
| `ALLM_BASE` | AnythingLLM 地址（遗留方案） | 默认 `http://127.0.0.1:3001` |
| `ALLM_KEY` | AnythingLLM API Key（遗留方案） | — |
| `ALLM_SLUG` | AnythingLLM 工作区 slug（遗留方案） | 默认 `scenic-spots` |
| `TTS_VOICE` | Edge-TTS 音色（遗留方案） | 默认 `zh-CN-YunxiNeural` |

## 测试与质检

```bash
python -m unittest tests.test_preprocess tests.test_run_e2e tests.test_server tests.test_extract   # 61 项单元测试
python -m ruff check .                                        # 代码质量
```

## 效果评测（RAG）

内置 29 题评测集（覆盖当前知识库 11 个景点 × 门票/开放时间/交通/看点/历史等字段 + 2 条库外负样本），一键跑 3 轮并生成报告：

```bash
python scripts/evaluate_rag.py --round 1   # 跑第 1 轮（检索 + 端到端问答）
python scripts/evaluate_rag.py --round 2   # 第 2 轮（独立采样）
python scripts/evaluate_rag.py --round 3   # 第 3 轮
python scripts/evaluate_rag.py --report    # 生成 eval/REPORT.md（3 轮汇总 + 逐题明细）
python scripts/evaluate_rag.py --summary   # 控制台汇总
python scripts/evaluate_rag.py --no-llm    # 只测检索层，不调用 LLM（省成本）
```

**当前结果**（2026-08-19，详见 `eval/REPORT.md`）：

| 指标 | 3 轮均值 |
|---|---|
| 检索 Recall@4 / 阈值命中 / Top1 | **100% / 100% / 100%** |
| 库外问题拒绝率 | **100%** |
| 问答严格通过率（自动初判） | **~65%** |
| 问答基本正确率（PASS+PARTIAL） | **~90%** |

> 说明：问答自动判定为关键词命中口径，偏严；`eval/results/roundN.csv` 可人工复核（在 judge 列旁加 human 列）。
> 评测已知缺陷：门票类问答存在 LLM 数字幻觉（如长隆"约 300 元"被编造为精确票价），已通过系统提示词约束缓解，未根除。

## AnythingLLM 部署（遗留方案，未维护）

> 当前主线为「直连版」（零 Docker）。以下 AnythingLLM + Docker 方案仅作为完整 RAG 管理台的替代选项保留，代码中的 `examples/demo_*.py` 与 `docs/最优配置方案.md` 部分章节与之对应，不再主动维护。

```bash
cd deploy
docker compose --env-file ../.env up -d
# 访问 http://127.0.0.1:3001（密码见 docker-compose 的 AUTH_TOKEN）
```

> 需本机已装 Docker Desktop。

## 知识库文档格式

每个景点一个 `## 景点名` 块，固定字段：

```
## 广州塔
简介：...
历史：...
文化：...
看点：...
交通：...
开放时间：...
门票：...
贴士：...
```

修改 `sample-data/` 后，重启服务会自动增量嵌入新文档；详见 `scripts/preprocess.py`。

# 景点讲解 AI（AnythingLLM）

基于 AnythingLLM 的低延迟、高准确度实时景点讲解系统。方案细节见 `docs/最优配置方案.md`。

## 目录结构

```
├── docs/最优配置方案.md       # 完整方案(选型/分块/流式/缓存/预处理/参数表)
├── deploy/docker-compose.yml  # AnythingLLM 一键部署
├── web/
│   ├── server.py              # 网页后端(标准库,代理流式+缓存)
│   └── index.html             # 景点讲解网页(流式+语音)
├── scripts/
│   ├── preprocess.py          # 景点文档清洗+结构化分块
│   ├── demo_chat.py           # 流式文本讲解客户端
│   └── demo_voice.py          # 流式语音讲解客户端(TTS)
└── sample-data/景点示例.md     # 结构化景点知识示例
```

## 快速开始

```bash
# 1. 申请两个 Key, 通过环境变量或 .env 传入(勿写入仓库):
#    - DEEPSEEK_API_KEY                 https://platform.deepseek.com
#    - ZHIPU_API_KEY (智谱 embedding)   https://open.bigmodel.cn
cp .env.example .env   # 填入两个 Key

# 2. 启动 AnythingLLM (读取 .env 中的 Key)
cd deploy
docker compose --env-file ../.env up -d
# 访问 http://localhost:3001 (密码见 docker-compose 的 AUTH_TOKEN)

# 3. 预处理景点文档
python scripts/preprocess.py 你的景点文档目录 -o sample-data

# 4. 网页上传 sample-data 到工作区 → 「保存并嵌入」→ 开始讲解
```

## Web 网页（面向游客）

```bash
# 启动网页后端(纯标准库, 零第三方依赖)
python web/server.py
# 浏览器打开 http://localhost:8080
```

网页功能：景点卡片选择、流式讲解、语音朗读、引用来源。配置在 `web/server.py` 顶部常量。

## 客户端调用（命令行）

```bash
pip install requests edge-tts

# 流式文本讲解
export ALLM_KEY="你的API Key"    # AnythingLLM 设置页 → API Keys 生成
python scripts/demo_chat.py "介绍一下陈家祠的历史"

# 流式语音讲解(生成 讲解音频.mp3)
python scripts/demo_voice.py "给我讲讲广州塔"
```

## 核心选型（速记）

| 组件 | 选型 | 理由 |
|---|---|---|
| 嵌入模型 | **智谱 embedding-3**（云端） | 中文原生, 2048 维, 无需本地 Ollama |
| 聊天模型 | **deepseek-chat**（云端） | 中文强+流式快+极低成本 |
| 向量库 | **LanceDB** | 内嵌零配置, 本地最快 |
| 分块 | 500~800 字符 / 50 重叠 | 按景点条目语义切分 |
| 流式 | stream-chat + 逐句语音 | 首字即回传, 首句即开播 |
| 缓存 | 语义响应缓存 + 本地索引 | 消除重复请求开销 |

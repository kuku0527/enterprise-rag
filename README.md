# Enterprise RAG System

企业级检索增强生成（RAG）系统，用于大模型工程师求职项目展示。

## 项目特点

- **完整的 RAG 流程**：文档加载 → 文本分块 → 向量化 → 检索 → 重排序 → 问答
- **多格式支持**：PDF、DOCX、TXT 文档
- **多种分块策略**：固定大小、句子级、语义级分块
- **高效向量检索**：基于 FAISS 的向量数据库
- **智能重排序**：使用 BGE Reranker 提升检索质量
- **REST API**：基于 FastAPI 的高性能接口
- **易于扩展**：模块化设计，方便定制

## 项目结构

```
enterprise-rag/
├── main.py                 # 主入口（API/CLI）
├── requirements.txt        # 依赖包
├── .env.example           # 环境变量模板
├── .gitignore             # Git 忽略文件
│
├── configs/               # 配置文件
│   ├── __init__.py
│   └── settings.py        # 系统配置
│
├── loader/                # 文档加载器
│   ├── __init__.py
│   └── document_loader.py # 多格式文档加载
│
├── parser/                # 文本解析器
│   ├── __init__.py
│   └── text_parser.py     # 文本分块策略
│
├── vectorstore/           # 向量数据库
│   ├── __init__.py
│   └── faiss_store.py     # FAISS 向量存储
│
├── service/               # 核心服务
│   ├── __init__.py
│   ├── embedding_service.py   # Embedding 服务
│   ├── reranker_service.py    # 重排序服务
│   └── rag_service.py         # RAG 主服务
│
├── data/                  # 示例数据
│   └── example.txt
│
└── tests/                 # 测试用例
    └── test_rag.py
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/enterprise-rag.git
cd enterprise-rag

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，配置以下变量：
# - OPENAI_API_KEY（可选，用于生成回答）
# - EMBEDDING_MODEL（默认使用 BAAI/bge-small-zh-v1.5）
# - 其他参数根据需要调整
```

### 3. 准备数据

将您的文档放入 `data/` 目录：
- 支持格式：PDF、DOCX、TXT
- 可以创建子目录组织文档

### 4. 运行系统

#### 方式一：命令行交互模式（推荐新手）

```bash
python main.py --mode cli
```

#### 方式二：REST API 模式

```bash
python main.py --mode api --port 8000
```

API 文档地址：http://localhost:8000/docs

## API 使用说明

### 查询接口

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是 RAG？",
    "top_k": 5,
    "rerank_top_n": 3
  }'
```

### 文档导入接口

```bash
# 导入单个文件
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "data/example.txt",
    "is_directory": false
  }'

# 导入整个目录
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "data",
    "is_directory": true
  }'
```

### 文件上传接口

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"
```

## 核心组件说明

### 1. 文档加载器 (loader/)

支持多种文档格式：
- `TextLoader`: 纯文本文件 (.txt)
- `PDFLoader`: PDF 文档 (.pdf)
- `DocxLoader`: Word 文档 (.docx)

### 2. 文本解析器 (parser/)

三种分块策略：
- `TextParser`: 固定大小分块，支持重叠
- `SentenceParser`: 基于句子边界分块
- `SemanticParser`: 语义感知分块（基于段落）

### 3. 向量数据库 (vectorstore/)

基于 FAISS 的向量存储：
- 支持多种索引类型：Flat、IVF、HNSW
- 自动向量归一化，支持余弦相似度
- 支持持久化存储和加载

### 4. Embedding 服务 (service/embedding_service.py)

两种 Embedding 方案：
- `EmbeddingService`: 使用 Sentence Transformers（本地模型）
- `OpenAIEmbeddingService`: 使用 OpenAI API

### 5. 重排序服务 (service/reranker_service.py)

两种重排序方案：
- `RerankerService`: 使用 BGE Reranker
- `CrossEncoderReranker`: 使用 Cross-Encoder

## 配置说明

在 `configs/settings.py` 中可以调整以下参数：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| CHUNK_SIZE | 文本块大小 | 512 |
| CHUNK_OVERLAP | 块重叠大小 | 50 |
| TOP_K | 检索文档数量 | 5 |
| USE_RERANKER | 是否使用重排序 | true |
| RERANKER_TOP_N | 重排序后保留数量 | 3 |

## 扩展指南

### 添加新的文档格式

1. 在 `loader/document_loader.py` 中创建新的 Loader 类
2. 继承 `BaseLoader` 并实现 `load` 和 `load_dir` 方法
3. 在 `DocumentLoader.LOADER_MAP` 中注册新格式

### 添加新的分块策略

1. 在 `parser/text_parser.py` 中创建新的 Parser 类
2. 实现 `parse` 方法，返回 `TextChunk` 列表

### 集成其他向量数据库

1. 在 `vectorstore/` 目录下创建新的 Store 类
2. 实现 `add_documents`、`search`、`save`、`load` 方法

### 集成其他 LLM

1. 在 `service/rag_service.py` 的 `_generate_answer` 方法中调用 LLM API
2. 支持 OpenAI、Anthropic、本地模型等

## 性能优化建议

1. **GPU 加速**：将 `EMBEDDING_DEVICE` 设置为 `cuda` 使用 GPU
2. **批量处理**：增大 `EMBEDDING_BATCH_SIZE` 提升吞吐量
3. **索引优化**：对于大数据集，使用 IVF 或 HNSW 索引
4. **缓存机制**：对频繁查询的结果进行缓存

## 常见问题

### Q: 模型下载失败怎么办？

A: 配置 Hugging Face 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 内存不足怎么办？

A: 
- 减小 `CHUNK_SIZE` 和 `EMBEDDING_BATCH_SIZE`
- 使用 `faiss-gpu` 替代 `faiss-cpu`
- 使用更小的 Embedding 模型

### Q: 如何提升检索质量？

A: 
- 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`
- 启用重排序（`USE_RERANKER=true`）
- 尝试不同的 Embedding 模型

## 技术栈

- **LLM Framework**: LlamaIndex
- **Vector Database**: FAISS
- **Embedding**: Sentence Transformers / OpenAI
- **Reranker**: BGE Reranker / Cross-Encoder
- **API Framework**: FastAPI
- **Document Processing**: PyPDF, python-docx

## License

MIT License

## 联系方式

如有问题，欢迎提 Issue 或联系作者。

# ⚡ RAG Backend
### Hybrid Retrieval + Generation + Evaluation + Feedback

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.133-green?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/ChromaDB-1.5-orange?style=flat&logo=chromadb&logoColor=white" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Claude-API-purple?style=flat&logo=anthropic&logoColor=white" alt="Claude">
  <img src="https://img.shields.io/badge/license-MIT-red?style=flat" alt="License">
  <img src="https://img.shields.io/badge/status-production--ready-brightgreen?style=flat" alt="Status">
</p>

---

## 📋 Содержание

- [Описание](#-описание)
- [Архитектура](#-архитектура)
- [Как это работает](#-как-это-работает)
- [Установка](#-установка)
- [Конфигурация](#-конфигурация)
- [API Endpoints](#-api-endpoints)
- [Технические детали](#-технические-детали)
- [Project Structure](#-project-structure)

---

## 🎯 Описание

**Production-ready** RAG (Retrieval-Augmented Generation) backend, который:

| 📥 Ingest | 🔍 Retrieve | 🤖 Generate | 📊 Evaluate | 💬 Feedback |
|-----------|------------|-------------|-------------|-------------|
| Загружай любые документы | Ищи релевантный контекст | Генерируй ответы по контексту | Оценивай качество ответов | Собирай фидбек и статистику |

Стек: **FastAPI** + **ChromaDB** + **sentence-transformers** + **Anthropic Claude** + **BM25**

---

## 🏗 Архитектура

```
                         ┌───────────┐
                         │ Document  │
                         └─────┬─────┘
                               │
                     ┌─────────▼─────────┐
                     │   chunk + embed   │
                     │  + ChromaDB+BM25  │
                     └─────────┬─────────┘
                               │
              ┌────────────────▼────────────────┐
              │   Hybrid Retrieval (RRF Fusion)  │
              │  ┌─────────┐    ┌──────────┐    │
              │  │  Dense  │    │   BM25   │    │
              │  │(Chroma) │  + │(keyword) │    │
              │  └────┬────┘    └────┬─────┘    │
              │       └──────┬──────┘           │
              │         ┌────▼────┐             │
              │         │ RRF(k=60) top-8       │
              │         └─────────┘             │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Claude Generation  │
                    │  (grounded answer)  │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │  Claude Judge (LLM-as-judge)     │
              │  relevance · completeness        │
              │  specificity · fluency           │
              └────────────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Feedback & Stats   │
                    └─────────────────────┘
```

### LLM Tiers

| Модель | Роль | Назначение |
|--------|------|------------|
| 🧠 `claude-sonnet-4-20250514` | **Генерация** | Формирует ответ по контексту |
| ⚖️ `claude-haiku-3-5-20241022` | **Оценка** | Судья — оценивает качество ответа |

---

## 🔄 Как это работает

### 1. Загрузка документа `POST /ingest`
```
Текст → tiktoken chunking (800 токенов, 120 overlap)
     → all-MiniLM-L6-v2 embedding (384d)
     → ChromaDB (dense index)
     → BM25 (keyword index)
```

### 2. Поиск `GET /retrieve` или `POST /query`
```
Запрос → dense search (семантический поиск)
       → BM25 search (ключевой поиск)
       → RRF Fusion (объединение рангов)
       → Top-K результатов
```

### 3. Генерация `POST /query`
```
Запрос + контекст → Claude → grounded answer
```

### 4. Оценка `POST /evaluate`
```
Запрос + ответ + контекст → Claude-судья → 4 метрики (1-5)
```

### 5. Фидбек `POST /feedback` + `GET /stats`
```
Оценки пользователя → агрегированная статистика
```

---

## ⚙️ Установка

```bash
# Клонируем
git clone <repo-url>
cd c:\

# Создаём виртуальное окружение (Windows)
python -m venv venv
venv\Scripts\activate

# Ставим зависимости
pip install -r requirements.txt

# Создаём .env с API ключом (см. ниже)

# Запускаем
python main.py
```

Сервер запустится на **`http://localhost:8000`**.
Документация Swagger: **`http://localhost:8000/docs`** 📖

---

## 🔧 Конфигурация

Создай `.env` в корне проекта:

```env
# 👉 Обязательно
ANTHROPIC_API_KEY=sk-ant-...

# 👉 Опционально (значения по умолчанию)
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=8
RRF_K=60
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=chroma_db/
```

---

## 🚀 API Endpoints

### 1️⃣ Загрузить документ

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Transformer architecture introduced the self-attention mechanism, which allows models to weigh the importance of different tokens in a sequence. This innovation became the foundation for modern LLMs like GPT, BERT, and Claude.",
    "source": "user upload"
  }'
```

📥 **Ответ:**
```json
{
  "doc_id": "a1b2c3d4-...",
  "chunk_count": 1,
  "message": "Ingested 1 chunks"
}
```

---

### 2️⃣ Поиск чанков

```bash
curl "http://localhost:8000/retrieve?q=transformer%20self-attention%20mechanism&top_k=5"
```

---

### 3️⃣ Сгенерировать ответ

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Transformer architecture and how does self-attention work?",
    "top_k": 5
  }'
```

📥 **Ответ:**
```json
{
  "query": "What is the Transformer architecture...",
  "generated_answer": "The Transformer architecture...",
  "sources": [
    {
      "chunk_id": "...",
      "text": "The Transformer architecture introduced...",
      "source": "",
      "score": 0.8942,
      "rank": 1
    }
  ]
}
```

---

### 4️⃣ Оценить ответ

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Transformer architecture?",
    "generated_answer": "...",
    "context": "..."
  }'
```

📥 **Ответ:**
```json
{
  "query": "What is the Transformer architecture?",
  "scores": [
    {"dimension": "relevance", "score": 4.5, "reasoning": "..."},
    {"dimension": "completeness", "score": 4.0, "reasoning": "..."},
    {"dimension": "specificity", "score": 3.5, "reasoning": "..."},
    {"dimension": "fluency", "score": 5.0, "reasoning": "..."}
  ],
  "average_score": 4.25
}
```

---

### 5️⃣ Отправить фидбек

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Transformer architecture?",
    "generated_answer": "...",
    "relevance_score": 4.5,
    "completeness_score": 4.0,
    "specificity_score": 3.5,
    "fluency_score": 5.0,
    "comment": "Very informative answer"
  }'
```

---

### 6️⃣ Статистика фидбека

```bash
curl http://localhost:8000/stats
```

📥 **Ответ:**
```json
{
  "total_feedback": 2,
  "avg_relevance": 3.75,
  "avg_completeness": 3.5,
  "avg_specificity": 3.25,
  "avg_fluency": 4.5,
  "avg_overall": 3.75,
  "recent_entries": [...]
}
```

---

## 🛠 Технические детали

### Chunking

| Параметр | Значение | Описание |
|----------|----------|----------|
| Размер чанка | `800` токенов | tiktoken cl100k_base |
| Перекрытие | `120` токенов | 15% для связности контекста |

### Retrieval

| Компонент | Технология | Размерность |
|-----------|-----------|-------------|
| 🧬 Dense search | ChromaDB (cosine similarity) | 384d |
| 🔤 Keyword search | BM25 Okapi (rank_bm25) | — |
| ♻️ Fusion | RRF (k=60) → top-8 | — |

### Evaluation Dimensions

| Метрика | Что оценивает |
|----------|--------------|
| 🎯 **Relevance** | Насколько ответ соответствует запросу |
| 📋 **Completeness** | Все ли аспекты запроса покрыты |
| 🔬 **Specificity** | Уровень детализации (не generic) |
| 💬 **Fluency** | Естественность и читаемость языка |

---

## 📁 Project Structure

```
c:\photo_to_prompt/
│
├── main.py              # 🚀 FastAPI entry point
├── config.py            # ⚙️ Settings (pydantic-settings)
├── models.py            # 📦 Pydantic schemas
│
├── ingestion.py         # 📥 Chunk → Embed → ChromaDB + BM25
├── retrieval.py         # 🔍 Dense + BM25 + RRF hybrid search
├── generation.py        # 🤖 Claude answer generation
├── evaluation.py        # ⚖️ LLM-as-judge scoring
├── feedback.py          # 💬 Feedback store + stats
│
├── chroma_db/           # 🗄️ Persistent vector store
├── requirements.txt     # 📜 Dependencies
└── README.md            # 📖 This file
```

---

<p align="center">
  <b>
  <a href="http://localhost:8000/docs">📖 Swagger Docs</a> ·
  <a href="http://localhost:8000/redoc">📘 ReDoc</a>
  </b>
</p>

<p align="center">
  <i>Built with FastAPI · ChromaDB · sentence-transformers · Anthropic Claude</i>
</p>
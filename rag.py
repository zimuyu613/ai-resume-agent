import os
import re
import math
import hashlib
from pathlib import Path

# ChromaDB 的间接依赖可能会触发 protobuf descriptor 兼容问题。
# 必须在任何 chromadb 相关 import 之前设置该环境变量。
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from dotenv import load_dotenv
from google import genai


load_dotenv()

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "resume_chunks"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RESUME_TEXT_LIMIT = 8000
MAX_CHUNKS = 12
DEFAULT_TOP_K = 3

_local_bge_model = None


def get_embedding_provider() -> str:
    """读取当前 RAG 向量模式；默认 local，避免免费 Gemini Embedding 继续触发 429。"""
    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).strip().lower()
    return provider if provider in {"local", "local_bge", "gemini"} else "local"


def build_chunk_records(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[dict]:
    """
    将中英文简历文本切成带 metadata 的 chunk 记录。
    char_start/char_end 基于清洗后的文本位置，用于解释 chunk 来源范围。
    """
    cleaned_text = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    overlap = max(0, min(overlap, chunk_size - 1))
    records: list[dict] = []
    start = 0

    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunk = cleaned_text[start:end]

        # 尽量在中英文标点或空格处收尾，让召回片段更自然。
        if end < len(cleaned_text):
            split_at = max(
                chunk.rfind("。"),
                chunk.rfind("！"),
                chunk.rfind("？"),
                chunk.rfind("."),
                chunk.rfind("!"),
                chunk.rfind("?"),
                chunk.rfind(";"),
                chunk.rfind("；"),
                chunk.rfind(" "),
            )
            if split_at > chunk_size * 0.5:
                end = start + split_at + 1
                chunk = cleaned_text[start:end]

        chunk = chunk.strip()
        if chunk:
            records.append(
                {
                    "text": chunk,
                    "chunk_id": len(records),
                    "char_start": start,
                    "char_end": end,
                    "chunk_length": len(chunk),
                    "section": "unknown",
                }
            )

        if end >= len(cleaned_text):
            break

        start = max(end - overlap, start + 1)

    return records


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """
    将中英文简历文本切成适合向量检索的小片段。
    优先按句子边界切分，避免产生空字符串。
    """
    return [record["text"] for record in build_chunk_records(text, chunk_size, overlap)]


def _is_resource_exhausted_error(error: Exception) -> bool:
    """识别 Gemini API 免费额度或请求频率限制相关错误。"""
    error_text = str(error).lower()
    quota_keywords = [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "rate_limit",
        "requests per minute",
        "请求频率",
        "免费额度",
        "额度不足",
    ]
    return any(keyword in error_text for keyword in quota_keywords)


def get_local_embedding(text: str, dim: int = 384) -> list[float]:
    """
    使用本地 hash embedding 生成固定维度向量，不依赖外部 API。
    适合演示和 fallback：将中英文 token 哈希到向量桶，再做 L2 归一化。
    """
    if dim <= 0:
        raise ValueError("dim 必须大于 0")

    vector = [0.0] * dim
    normalized_text = (text or "").lower()
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", normalized_text)

    if not tokens:
        tokens = list(normalized_text.strip())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def _encode_local_bge_embedding(text: str) -> list[float]:
    """严格使用 local_bge 生成向量；失败时抛出异常，由调用方决定是否整批 fallback。"""
    global _local_bge_model

    if _local_bge_model is None:
        from sentence_transformers import SentenceTransformer

        print(f"正在加载本地语义 embedding 模型：{LOCAL_EMBEDDING_MODEL}")
        _local_bge_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

    embedding = _local_bge_model.encode(
        text or "",
        normalize_embeddings=True,
    )

    return [float(value) for value in embedding.tolist()]


def get_local_bge_embedding(text: str) -> list[float]:
    """
    使用 sentence-transformers 本地语义模型生成向量。
    如果模型导入、下载、加载或推理失败，自动 fallback 到本地 hash embedding。
    """
    try:
        return _encode_local_bge_embedding(text)

    except Exception as e:
        print(f"local_bge embedding 加载或生成失败，已 fallback 到 local hash embedding：{e}")
        return get_local_embedding(text)


def _embed_text(text: str, task_type: str | None = None) -> list[float]:
    """
    调用 Gemini Embedding 模型。
    task_type 用于区分文档入库和查询检索；如果 SDK 不支持该参数，会自动降级。
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 GEMINI_API_KEY，请先在 .env 文件中配置 Gemini API Key。")

    client = genai.Client(api_key=api_key)

    try:
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config={"task_type": task_type} if task_type else None,
            )
        except TypeError:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )
    except Exception as e:
        if _is_resource_exhausted_error(e):
            raise RuntimeError(
                "Gemini Embedding 请求触发 429 RESOURCE_EXHAUSTED，"
                "可能是免费额度或请求频率限制。请稍后重试，或关闭 RAG 模式使用普通分析。"
            ) from e

        raise RuntimeError(f"Gemini Embedding 调用失败：{e}") from e

    embeddings = getattr(response, "embeddings", None)
    if not embeddings:
        raise RuntimeError("Gemini Embedding 模型没有返回向量结果。")

    values = getattr(embeddings[0], "values", None)
    if not values:
        raise RuntimeError("Gemini Embedding 返回结果中缺少 values 字段。")

    return list(values)


def get_embedding(text: str) -> list[float]:
    """
    对外暴露的通用文本向量化函数。
    默认使用 local；支持 local_bge 和 gemini，失败时 fallback 到 local。
    """
    provider = get_embedding_provider()

    if provider == "local":
        print("RAG embedding provider: local hash embedding")
        return get_local_embedding(text)

    if provider == "local_bge":
        print("RAG embedding provider: local_bge")
        return get_local_bge_embedding(text)

    try:
        return _embed_text(text, task_type=None)
    except RuntimeError as e:
        if _is_resource_exhausted_error(e):
            print("Gemini Embedding 额度或频率受限，已自动切换到本地 hash embedding fallback。")
        else:
            print(f"Gemini Embedding 调用失败，已自动切换到本地 hash embedding fallback：{e}")
        return get_local_embedding(text)


def _embed_texts_for_rag(
    texts: list[str],
    task_type: str,
    provider_override: str | None = None,
) -> tuple[list[list[float]], str]:
    """
    为一次 RAG 流程批量生成向量。
    如果 local_bge 或 Gemini 失败，整批切换到 local，避免同一 collection 混用不同维度。
    """
    provider = provider_override or get_embedding_provider()

    if provider == "local":
        return [get_local_embedding(text) for text in texts], "local"

    if provider == "local_bge":
        try:
            return [_encode_local_bge_embedding(text) for text in texts], "local_bge"
        except Exception as e:
            print(f"local_bge 批量生成失败，本次 RAG 已使用 local hash embedding：{e}")
            return [get_local_embedding(text) for text in texts], "local"

    try:
        return [_embed_text(text, task_type=task_type) for text in texts], "gemini"
    except RuntimeError as e:
        if _is_resource_exhausted_error(e):
            print("Gemini Embedding 额度或频率受限，本次 RAG 已使用本地 hash embedding fallback。")
        else:
            print(f"Gemini Embedding 批量生成失败，本次 RAG 已使用 local hash embedding：{e}")
        return [get_local_embedding(text) for text in texts], "local"


def _get_chroma_client():
    """延迟导入 ChromaDB，便于在依赖未安装时给出友好错误。"""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("未安装 chromadb，请先运行 pip install -r requirements.txt。") from exc

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_resume_collection(
    resume_text: str,
    source_name: str = "简历文本",
    max_chunks: int = MAX_CHUNKS,
    provider_override: str | None = None,
):
    """
    为当前简历重建本地 ChromaDB collection。
    每次分析前删除旧 collection，避免上一份简历的片段干扰当前结果。
    """
    # 限制进入 RAG 的简历长度，避免长简历一次分析触发过多 Embedding 调用。
    resume_text = (resume_text or "")[:RESUME_TEXT_LIMIT]
    chunk_records = build_chunk_records(resume_text)
    if not chunk_records:
        raise RuntimeError("简历文本为空，无法构建 RAG 检索库。")

    if len(chunk_records) > max_chunks:
        chunk_records = chunk_records[:max_chunks]

    chunks = [record["text"] for record in chunk_records]

    client = _get_chroma_client()

    # 只删除同名 collection，避免 Windows 下直接删除 sqlite 文件时遇到文件占用。
    # 这样仍然可以确保当前分析只使用当前简历片段。
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    embeddings, provider_used = _embed_texts_for_rag(
        chunks,
        task_type="RETRIEVAL_DOCUMENT",
        provider_override=provider_override,
    )

    collection.add(
        ids=[f"resume_chunk_{index}" for index in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "source": "resume",
                "source_name": source_name,
                "file_name": source_name,
                "chunk_index": index + 1,
                "chunk_id": record["chunk_id"],
                "char_start": record["char_start"],
                "char_end": record["char_end"],
                "chunk_length": record["chunk_length"],
                "section": record["section"],
                "embedding_provider": provider_used,
            }
            for index, record in enumerate(chunk_records)
        ],
    )

    return collection, provider_used


def retrieve_relevant_chunks_with_sources(
    job_description: str,
    resume_text: str,
    source_name: str = "简历文本",
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    根据岗位描述，从当前简历 collection 中召回最相关的片段。
    返回给模型使用的上下文，以及页面可展示的来源片段信息。
    """
    collection, provider_used = build_resume_collection(
        resume_text,
        source_name=source_name,
    )

    if provider_used == "local":
        query_embedding = get_local_embedding(job_description)
    elif provider_used == "local_bge":
        try:
            query_embedding = _encode_local_bge_embedding(job_description)
        except Exception as e:
            # 查询阶段 local_bge 失败时，重建本地 hash collection，确保查询和文档向量维度一致。
            print(f"local_bge 查询 embedding 失败，本次 RAG 已重建为 local hash embedding：{e}")
            collection, provider_used = build_resume_collection(
                resume_text,
                source_name=source_name,
                provider_override="local",
            )
            query_embedding = get_local_embedding(job_description)
    else:
        try:
            query_embedding = _embed_text(job_description, task_type="RETRIEVAL_QUERY")
        except RuntimeError as e:
            # 查询阶段 Gemini 失败时，重建本地 hash collection，确保查询和文档向量维度一致。
            if _is_resource_exhausted_error(e):
                print("Gemini 查询 embedding 触发限流，本次 RAG 已重建为本地 hash embedding。")
            else:
                print(f"Gemini 查询 embedding 失败，本次 RAG 已重建为本地 hash embedding：{e}")
            collection, provider_used = build_resume_collection(
                resume_text,
                source_name=source_name,
                provider_override="local",
            )
            query_embedding = get_local_embedding(job_description)

    result_count = min(max(top_k, 1), collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0] if results else []
    metadatas = results.get("metadatas", [[]])[0] if results else []
    distances = results.get("distances", [[]])[0] if results else []

    sources = []
    context_parts = []

    for index, document in enumerate(documents, start=1):
        text = (document or "").strip()
        if not text:
            continue

        metadata = metadatas[index - 1] if index - 1 < len(metadatas) else {}
        distance = distances[index - 1] if index - 1 < len(distances) else None
        chunk_index = metadata.get("chunk_index", index)
        chunk_id = metadata.get("chunk_id", chunk_index)
        source_display_name = metadata.get("source_name", source_name)
        file_name = metadata.get("file_name", source_display_name)

        context_parts.append(f"[相关片段 {index} | 来源：{source_display_name} | chunk_id：{chunk_id}]\n{text}")
        source_item = {
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "source_name": source_display_name,
            "source": metadata.get("source", "resume"),
            "file_name": file_name,
            "char_start": metadata.get("char_start"),
            "char_end": metadata.get("char_end"),
            "chunk_length": metadata.get("chunk_length", len(text)),
            "section": metadata.get("section", "unknown"),
            "text": text,
        }

        if distance is not None:
            source_item["distance"] = distance

        sources.append(source_item)

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
        "embedding_provider": provider_used,
        "total_chunks": collection.count(),
    }


def retrieve_relevant_chunks(job_description: str, resume_text: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    兼容旧调用：只返回拼接后的 RAG 上下文文本。
    """
    result = retrieve_relevant_chunks_with_sources(
        job_description=job_description,
        resume_text=resume_text,
        top_k=top_k,
    )

    return result["context"]

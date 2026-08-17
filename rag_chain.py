"""LangChain RAG chain for KYK regulation QA using DeepSeek and ChromaDB."""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _env_float(name, default):
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc


DEEPSEEK_INPUT_PRICE = _env_float("DEEPSEEK_INPUT_PRICE_PER_M", 0.14)
DEEPSEEK_OUTPUT_PRICE = _env_float("DEEPSEEK_OUTPUT_PRICE_PER_M", 0.28)

SYSTEM_PROMPT = """Sen bir KYK (Kredi ve Yurtlar Kurumu) mevzuat ve yurt hizmetleri yönetmeliği uzmanısın.
Aşağıda sana verilen bağlam, KYK yönetmeliğinin ilgili bölümlerinden alınmıştır.

Kurallar:
1. YALNIZCA verilen bağlamdaki resmi hükümlere dayanarak cevap ver.
2. Bağlamda cevabı bulunmayan sorular için tahmin yürütme; "Bu konuda yönetmelikte bir bilgi bulunmamaktadır." de.
3. Cevabında yararlandığın bölüm adını veya etiketi açıkça belirt.
4. Cevapları net, resmi ve anlaşılır biçimde düzenle.
5. İlgili madde numaralarını (ör. Madde 6, Madde 18) belirt.

Bağlam:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Soru: {question}\nCevap:"),
    ]
)


def _require_key(value, env_name):
    if not value or value.startswith("your_"):
        raise RuntimeError(f"{env_name} is not configured. Copy .env.example to .env and set it.")
    return value


def get_vector_store():
    openai_key = _require_key(OPENAI_API_KEY, "OPENAI_API_KEY")
    if not os.path.isdir(CHROMA_PATH):
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_PATH}. Run: python embed_summaries.py"
        )
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=openai_key
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="kyk_summaries",
    )


def _doc_identity(doc):
    metadata = doc.metadata or {}
    return (
        metadata.get("record_id")
        or metadata.get("etiket")
        or metadata.get("subject")
        or doc.page_content
    )


def deduplicate_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        identity = _doc_identity(doc)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(doc)
    return unique


def format_docs(docs):
    parts = []
    for index, doc in enumerate(deduplicate_docs(docs), 1):
        metadata = doc.metadata or {}
        etiket = metadata.get("etiket", "")
        subject = metadata.get("subject", "")
        markdown = metadata.get("markdown_content", "")
        parts.append(
            f"--- Bölüm {index}: {etiket} ({subject}) ---\n"
            f"İçerik:\n{markdown}\n"
        )
    return "\n".join(parts)


def get_llm():
    deepseek_key = _require_key(DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=deepseek_key,
        temperature=0.2,
        max_tokens=4096,
    )


def _token_usage(ai_message):
    usage_metadata = getattr(ai_message, "usage_metadata", None) or {}
    token_usage = (ai_message.response_metadata or {}).get("token_usage", {})

    input_tokens = usage_metadata.get("input_tokens")
    if input_tokens is None:
        input_tokens = token_usage.get("prompt_tokens", 0)

    output_tokens = usage_metadata.get("output_tokens")
    if output_tokens is None:
        output_tokens = token_usage.get("completion_tokens", 0)

    total_tokens = usage_metadata.get("total_tokens")
    if total_tokens is None:
        total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)

    return int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0)


def ask_with_sources(query: str, k: int = 5):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = deduplicate_docs(retriever.invoke(query.strip()))
    if not docs:
        raise RuntimeError("No matching regulation sections were found in ChromaDB.")

    context = format_docs(docs)
    ai_message = (prompt | get_llm()).invoke(
        {"context": context, "question": query.strip()}
    )
    answer = ai_message.content

    input_tokens, output_tokens, total_tokens = _token_usage(ai_message)
    cost_usd = (
        (input_tokens / 1_000_000) * DEEPSEEK_INPUT_PRICE
        + (output_tokens / 1_000_000) * DEEPSEEK_OUTPUT_PRICE
    )

    return answer, docs, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }

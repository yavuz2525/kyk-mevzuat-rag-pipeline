"""
LangChain RAG Chain for KYK Dormitory Regulation QA using DeepSeek & ChromaDB.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# ── API Configurations ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── Pricing (per 1M tokens) ────────────────────────────────────────
DEEPSEEK_INPUT_PRICE = 0.14    # $ per 1M input tokens
DEEPSEEK_OUTPUT_PRICE = 0.28   # $ per 1M output tokens

# ── System Prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """Sen bir KYK (Kredi ve Yurtlar Kurumu) mevzuat ve yurt hizmetleri yönetmeliği uzmanısın.
Aşağıda sana verilen bağlam (context), KYK yönetmeliğinin ilgili bölümlerinden ve Teable bilgi tabanından alınmıştır.

Kurallar:
1. YALNIZCA verilen bağlamdaki resmi hükümlere dayanarak cevap ver.
2. Bağlamda cevabı bulunmayan sorular için kesinlikle tahmin yürütme ve "Bu konuda yönetmelikte bir bilgi bulunmamaktadır." şeklinde belirt.
3. Cevabında mutlaka hangi bölümden yararlandığını (Bölüm Adı veya Etiket) açıkça belirt.
4. Cevapları maddeler halinde, net, resmi ve anlaşılır bir dille düzenle.
5. İlgili Madde numaralarını (ör. Madde 6, Madde 18) kesinlikle belirt.

Bağlam (KYK Yönetmeliği İlgili Bölümleri):
{context}

Soru: {question}
Cevap:"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
])


def get_vector_store():
    """Lazy initialization of ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="kyk_summaries",
    )


def format_docs(docs):
    """Format retrieved documents into a single context string, deduplicating by record_id."""
    seen_ids = set()
    parts = []
    idx = 0
    for doc in docs:
        record_id = doc.metadata.get("record_id", "")
        if record_id in seen_ids:
            continue
        seen_ids.add(record_id)
        idx += 1
        etiket = doc.metadata.get("etiket", "")
        subject = doc.metadata.get("subject", "")
        markdown = doc.metadata.get("markdown_content", "")
        parts.append(
            f"--- Bölüm {idx}: {etiket} ({subject}) ---\n"
            f"İçerik:\n{markdown}\n"
        )
    return "\n".join(parts)


def deduplicate_docs(docs):
    """Remove duplicate documents while preserving relevance order."""
    seen_ids = set()
    unique = []
    for doc in docs:
        record_id = doc.metadata.get("record_id", "")
        if record_id not in seen_ids:
            seen_ids.add(record_id)
            unique.append(doc)
    return unique


def get_llm():
    """Initialize DeepSeek Chat LLM."""
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        temperature=0.2,
        max_tokens=4096,
    )


def ask_with_sources(query: str, k: int = 5):
    """
    Runs the full RAG pipeline:
    1. Retrieves similar summaries from ChromaDB
    2. Injects full markdown context
    3. Generates structured answer via DeepSeek LLM
    4. Calculates token usage and estimated cost

    Returns:
        answer (str), docs (list), token_info (dict)
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(query)
    docs = deduplicate_docs(docs)
    context = format_docs(docs)

    llm = get_llm()
    chain = prompt | llm

    ai_message = chain.invoke({"context": context, "question": query})
    answer = ai_message.content

    # Calculate token usage
    usage = ai_message.response_metadata.get("token_usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    cost_usd = (
        (input_tokens / 1_000_000) * DEEPSEEK_INPUT_PRICE +
        (output_tokens / 1_000_000) * DEEPSEEK_OUTPUT_PRICE
    )

    token_info = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }

    return answer, docs, token_info

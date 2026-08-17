"""Streamlit web interface for the KYK regulation RAG query system."""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="KYK Yönetmelik RAG Sistemi",
    page_icon="🏛️",
    layout="wide",
)

with st.sidebar:
    st.header("⚙️ RAG Ayarları")
    k = st.slider("Getirilecek Bölüm Sayısı (Top-k)", min_value=1, max_value=8, value=4)

    st.divider()
    st.subheader("🤖 Model Bilgileri")
    llm_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    st.markdown(f"**LLM:** `{llm_model}`")
    st.markdown("**Embedding:** `text-embedding-3-small`")
    st.markdown("**Vektör veritabanı:** `ChromaDB`")

    st.divider()
    st.caption("KYK Yurt Hizmetleri Yönetmeliği Mevzuat Asistanı")

st.title("🏛️ KYK Yönetmelik Soru-Cevap & RAG Sistemi")
st.markdown(
    "Bu sistem, **DeepSeek-OCR** ile yapılandırılan KYK Yurt Hizmetleri "
    "Yönetmeliği üzerinden soruları RAG yaklaşımıyla yanıtlar."
)

example_queries = [
    "Yurt ücreti süresinde ödenmezse hangi işlemler uygulanır?",
    "Öğrencilerin yurtta bir öğretim yılında kaç gün izin hakkı vardır?",
    "Hangi eylemler yurttan çıkarma cezası gerektirir?",
    "Yurtta kalma süresi nasıl belirlenir, uzatılabilir mi?",
]

selected_example = st.selectbox(
    "💡 Örnek Sorular (Seçip deneyebilirsiniz):",
    ["Kendi sorunuzu yazın..."] + example_queries,
)
default_text = "" if selected_example == "Kendi sorunuzu yazın..." else selected_example

query = st.text_input(
    "🔍 Mevzuatla ilgili sorunuzu girin:",
    value=default_text,
    placeholder="Örn: Yurt ücreti ve güvence bedeli ne zaman iade edilir?",
)

col1, col2, _ = st.columns([1, 1, 4])
with col1:
    submit = st.button("🚀 Soruyu Yanıtla", type="primary", use_container_width=True)
with col2:
    clear = st.button("🗑️ Temizle", use_container_width=True)

if clear:
    for key in ("answer", "sources", "token_info"):
        st.session_state.pop(key, None)
    st.rerun()

if submit and query.strip():
    with st.spinner("📡 İlgili mevzuat bölümleri taranıyor ve yanıt hazırlanıyor..."):
        try:
            from rag_chain import ask_with_sources

            answer, sources, token_info = ask_with_sources(query.strip(), k=k)
            st.session_state["answer"] = answer
            st.session_state["sources"] = sources
            st.session_state["token_info"] = token_info
        except Exception as exc:
            st.error(f"❌ Sorgulama sırasında hata oluştu: {exc}")
            st.info(
                "💡 Önce `python embed_summaries.py` komutunu çalıştırın ve "
                "`.env` dosyanızdaki API anahtarlarını kontrol edin."
            )

if "answer" in st.session_state:
    st.divider()
    st.subheader("📑 Resmi Mevzuata Dayalı Yanıt")
    st.markdown(st.session_state["answer"])

    token_info = st.session_state.get("token_info", {})
    if token_info:
        st.divider()
        st.subheader("📊 Token & Maliyet Analizi")

        try:
            usd_try_rate = float(os.getenv("USD_TRY_RATE", "") or 0)
        except ValueError:
            usd_try_rate = 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📥 Giriş Token", f"{token_info['input_tokens']:,}")
        m2.metric("📤 Çıkış Token", f"{token_info['output_tokens']:,}")
        m3.metric("🔢 Toplam Token", f"{token_info['total_tokens']:,}")

        cost_usd = token_info["cost_usd"]
        if usd_try_rate > 0:
            m4.metric(
                "💰 Tahmini Maliyet",
                f"${cost_usd:.6f}",
                f"~₺{cost_usd * usd_try_rate:.4f}",
            )
        else:
            m4.metric("💰 Tahmini Maliyet", f"${cost_usd:.6f}")

    sources = st.session_state.get("sources", [])
    if sources:
        st.divider()
        st.subheader("📚 Kullanılan Kaynak Mevzuat Bölümleri")
        for index, doc in enumerate(sources, 1):
            metadata = doc.metadata or {}
            tag = metadata.get("etiket", "Bölüm")
            subject = metadata.get("subject", "")
            with st.expander(f"📌 {index}. {tag} — {subject}"):
                st.markdown(metadata.get("markdown_content", "*İçerik bulunamadı*"))

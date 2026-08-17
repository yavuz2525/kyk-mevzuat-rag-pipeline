"""
Streamlit Web Interface for KYK Regulation RAG Query System.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="KYK Yönetmelik RAG Sistemi",
    page_icon="🏛️",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ RAG Ayarları")
    k = st.slider("Getirilecek Bölüm Sayısı (Top-k)", min_value=1, max_value=8, value=4)
    
    st.divider()
    st.subheader("🤖 Model Bilgileri")
    llm_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    st.markdown(f"**LLM:** `{llm_model}`")
    st.markdown("**Embedding:** `text-embedding-3-small`")
    st.markdown("**Veritabanı:** `ChromaDB + Teable`")

    st.divider()
    st.caption("KYK Yurt Hizmetleri Yönetmeliği Mevzuat Asistanı")

# ── Main Header ────────────────────────────────────────────────────
st.title("🏛️ KYK Yönetmelik Soru-Cevap & RAG Sistemi")
st.markdown("Bu sistem, **DeepSeek-OCR** ile taranıp yapılandırılan ve **Teable** bilgi tabanına aktarılan KYK Yurt Hizmetleri Yönetmeliği üzerinden sorularınızı yanıtlar.")

# ── Example Questions ──────────────────────────────────────────────
example_queries = [
    "Yurt ücreti süresinde ödenmezse hangi işlemler uygulanır?",
    "Öğrencilerin yurtta bir öğretim yılında kaç gün izin hakkı vardır?",
    "Hangi eylemler yurttan çıkarma cezası gerektirir?",
    "Yurtta kalma süresi nasıl belirlenir, uzatılabilir mi?"
]

selected_example = st.selectbox(
    "💡 Örnek Sorular (Seçip deneyebilirsiniz):",
    ["Kendi sorunuzu yazın..."] + example_queries
)

default_text = "" if selected_example == "Kendi sorunuzu yazın..." else selected_example

# ── Query Input ────────────────────────────────────────────────────
query = st.text_input(
    "🔍 Mevzuatla ilgili sorunuzu girin:",
    value=default_text,
    placeholder="Örn: Yurt ücreti ve güvence bedeli ne zaman iade edilir?"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    submit = st.button("🚀 Soruyu Yanıtla", type="primary", use_container_width=True)
with col2:
    clear = st.button("🗑️ Temizle", use_container_width=True)

if clear:
    st.session_state.pop("answer", None)
    st.session_state.pop("sources", None)
    st.session_state.pop("token_info", None)
    st.rerun()

# ── Query Execution ────────────────────────────────────────────────
if submit and query.strip():
    with st.spinner("📡 İlgili mevzuat bölümleri taranıyor ve DeepSeek yanıt hazırlıyor..."):
        try:
            from rag_chain import ask_with_sources
            answer, sources, token_info = ask_with_sources(query.strip(), k=k)
            st.session_state["answer"] = answer
            st.session_state["sources"] = sources
            st.session_state["token_info"] = token_info
        except Exception as e:
            st.error(f"❌ Sorgulama sırasında hata oluştu: {e}")
            st.info("💡 İpucu: Vektör veritabanını oluşturmak için önce `python embed_summaries.py` komutunu çalıştırdığınızdan ve `.env` dosyanızda API anahtarlarının tanımlı olduğundan emin olun.")

# ── Display Results ────────────────────────────────────────────────
if "answer" in st.session_state:
    st.divider()
    st.subheader("📑 Resmi Mevzuata Dayalı Yanıt")
    st.markdown(st.session_state["answer"])

    # Metrics
    ti = st.session_state.get("token_info", {})
    if ti:
        st.divider()
        st.subheader("📊 Token & Maliyet Analizi")
        usd_rate = 40.0  # Yaklaşık kur
        cost_tl = ti["cost_usd"] * usd_rate
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📥 Giriş Token", f"{ti['input_tokens']:,}")
        m2.metric("📤 Çıkış Token", f"{ti['output_tokens']:,}")
        m3.metric("🔢 Toplam Token", f"{ti['total_tokens']:,}")
        m4.metric("💰 Tahmini Maliyet", f"${ti['cost_usd']:.6f}", f"~₺{cost_tl:.4f}")

    # Sources
    sources = st.session_state.get("sources", [])
    if sources:
        st.divider()
        st.subheader("📚 Kullanılan Kaynak Mevzuat Bölümleri")
        for i, doc in enumerate(sources, 1):
            tag = doc.metadata.get("etiket", "Bölüm")
            subj = doc.metadata.get("subject", "")
            with st.expander(f"📌 {tag} — {subj}"):
                st.markdown(doc.metadata.get("markdown_content", "*İçerik bulunamadı*"))

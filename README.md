# 🏛️ DeepSeek-OCR ile Mevzuat RAG & Yapılandırılmış Bilgi Tabanı Pipeline'ı

> Karmaşık mevzuat ve hukuki belgeleri (**KYK Yurt Hizmetleri Yönetmeliği**) görme-dil modeli (**DeepSeek-OCR**) ile ayrıştıran, **Teable** üzerinde yapılandırılmış bilgi tabanına dönüştüren, **ChromaDB + DeepSeek LLM** ile sorgulayan ve **Streamlit Web Arayüzü** sunan uçtan uca RAG sistemi.

---

## 📌 Projeye Genel Bakış

Geleneksel metin çıkarıcılar (PyPDF, pdfplumber vb.), mevzuat metinlerindeki hiyerarşik başlıkları, dipnotları ve tabloları okurken yapısal kayıplara yol açar; cümleler sayfa sınırlarında bölünür ve anlamsal bütünlük bozulur.

Bu projede **Gelişmiş Görme Tabanlı RAG (Vision-RAG Document & QA Pipeline)** uygulanmıştır:
1. **Vision-to-Markdown OCR**: **DeepSeek-OCR** (4-bit BitsAndBytes kuantizasyonu) ile PDF sayfaları doğrudan yapılandırılmış Markdown ve HTML formatına dönüştürülür.
2. **Görsel Konumlandırma & Bounding Box**: Metin içi koordinatlar (`<|ref|>...<|det|>`) taranarak sayfa üzerindeki başlık, tablo ve figürler sınır kutularıyla işaretlenir (`_boxed.png`).
3. **Akıllı Satır Onarımı**: Sayfa sınırlarında bölünen cümleler ve paragraflar kurallara göre birleştirilir (`merge_and_split.py`).
4. **Mevzuat Odaklı Semantik Bölümleme**: Yönetmelik hiyerarşik 8 ana bölüme ayrılır.
5. **Teable Yapılandırılmış Bilgi Tabanı**: Bölümler; konu başlığı, metadata etiketi, maddeler özeti ve tam markdown içeriği ile Teable'a aktarılır.
6. **Vektörleştirme (ChromaDB)**: Bölüm özetleri `text-embedding-3-small` ile vektörleştirilir, metadatalara tam Markdown iliştirilir.
7. **RAG Soru-Cevap & Maliyet Takibi**: Gelen soruya en uygun bölümler çekilip **DeepSeek LLM (`deepseek-chat`)** modeline gönderilir, token ve maliyet hesabı yapılır.
8. **Streamlit Web UI**: Kullanıcı dostu web arayüzü ile interaktif sorgulama sağlanır.

---

## 🏗️ Uçtan Uca Mimari Akış Şeması

```text
  [KYK Yönetmeliği PDF (7.5.38510.pdf)]
                  │
                  ▼ (144 DPI Sayfa Render - PyMuPDF)
       [Yüksek Çözünürlüklü Sayfa Görselleri]
                  │
                  ▼ (DeepSeek-OCR Vision LLM / 4-bit NF4)
       [Markdown + Bounding Box Koordinatları + Görseller]
                  │
                  ▼ (merge_and_split.py: Satır Onarımı ve Metin Birleştirme)
       [Bütünleşik & Onarılmış Metin (merged_all_pages.md)]
                  │
                  ▼ (Hiyerarşik Bölüm & Madde Ayrıştırma)
       [Semantik Bölümler / Topics (00_Kapak ... 07_Yedinci_Bolum)]
                  │
                  ▼ (sync_to_teable.py / REST API veya Doğrudan DB)
       [Teable Bilgi Tabanı (Metadata + Özet + Tam Markdown)]
                  │
                  ▼ (embed_summaries.py / OpenAI Embeddings)
       [ChromaDB Vektör Veritabanı (kyk_summaries)]
                  │
                  ▼ (rag_chain.py / LangChain + DeepSeek LLM)
       [Streamlit Web UI (app.py) - İnteraktif Soru-Cevap & Maliyet Takibi]
```

---

## 📂 Dizin ve Dosya Yapısı

```text
.
├── 7.5.38510.pdf                  # Kaynak KYK Yurt Hizmetleri Yönetmeliği PDF'i
├── requirements.txt               # Proje bağımlılıkları (PyTorch, LangChain, Streamlit vb.)
├── .env.example                   # API anahtarları ve Teable konfigürasyon şablonu
├── .gitignore                     # Git hariç tutma kuralları
│
├── app.py                         # 🌐 Streamlit Web Arayüzü
├── rag_chain.py                   # 🧠 LangChain RAG Zinciri (DeepSeek LLM + ChromaDB)
├── embed_summaries.py             # ⚡ Özetleri Vektörleştirme ve ChromaDB'ye Kaydetme
├── query_embeddings.py            # 💻 Terminal üzerinden vektör arama CLI aracı
│
├── run_ocr_pages.py               # DeepSeek-OCR sayfa çıkarma pipeline'ı (4-bit kuantize)
├── merge_and_split.py             # Sayfa birleştirme, satır onarma ve semantik bölümleyici
├── sync_to_teable.py              # Teable API aktarımı ve JSON/CSV export scripti
│
├── teable_exported_data.json      # 8 semantik bölümün hazır veri seti (JSON)
├── teable_exported_data.csv       # 8 semantik bölümün hazır veri seti (CSV)
│
├── output/                        # İşlem çıktıları
│   ├── merged_all_pages.md        # Birleştirilmiş ve onarılmış tam markdown metni
│   ├── page_1 ... page_14/        # Sayfa bazlı Markdown, HTML, orijinal ve kutucuklu PNG'ler
│   └── topics/                    # Semantik olarak ayrıştırılmış bölümler (00_Kapak - 07_Yedinci_Bolum)
│
└── DeepSeek-OCR/                  # DeepSeek-OCR model mimarisi ve işleme kütüphanesi
```

---

## 📊 Teable Bilgi Tabanı Şeması

| Alan Adı | Veri Tipi | RAG Pipeline'ındaki Rolü |
| :--- | :--- | :--- |
| **`id`** | AutoNumber / Int | Benzersiz bölüm / chunk kimliği |
| **`Subject_String`** | SingleLineText | Bölüm başlığı ve kapsamı |
| **`Etiket`** | SingleLineText | Metadata etiketi (Örn: `01_Birinci_Bolum`, `06_Altinci_Bolum`) |
| **`summary`** | LongText | İlgili bölümün kapsadığı maddelerin özeti (Hızlı vektör arama) |
| **`markdown_content`** | LongText | OCR ile üretilmiş tam Markdown metni (LLM'e inject edilen Context) |

---

## 📑 Semantik Bölümleme Listesi (Knowledge Chunks)

1. **`00_Kapak`**: Yönetmelik Başlığı ve Giriş
2. **`01_Birinci_Bolum`**: Amaç, Kapsam, Dayanak ve Tanımlar (Madde 1-4)
3. **`02_Ikinci_Bolum`**: Yurt İhtiyacı, Barınma Şartları, Değerlendirme ve Yerleştirme İşlemleri (Madde 5-10)
4. **`03_Ucuncu_Bolum`**: Yurt Ücreti ve Güvence Bedeli (Madde 11-14)
5. **`04_Dorduncu_Bolum`**: Yurtların Açılması, Kapatılması ve Faaliyet Zamanları (Madde 15-16)
6. **`05_Besinci_Bolum`**: Barınma Süreleri ve İzinler (Madde 17-20)
7. **`06_Altinci_Bolum`**: Disiplin İşlemleri (Madde 21-34)
8. **`07_Yedinci_Bolum`**: Çeşitli ve Son Hükümler (Madde 35-47, Ek Madde 1, Geçici Maddeler)

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum ve Bağımlılıklar

```bash
# Repoyu klonlayın
git clone https://github.com/yavuz2525/kyk-mevzuat-rag-pipeline.git
cd kyk-mevzuat-rag-pipeline

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Ortam Değişkenlerini Ayarlayın

`.env.example` dosyasını `.env` olarak kopyalayın ve API anahtarlarınızı girin:
```bash
cp .env.example .env
```
`.env` içeriği:
```env
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### 3. Vektör Veritabanını Oluşturun (ChromaDB)

Hazır veri setinden (veya Teable'dan) özetleri vektörleştirip ChromaDB'ye kaydetmek için:
```bash
python embed_summaries.py
```

### 4. Web Arayüzünü Başlatın (Streamlit)

```bash
streamlit run app.py
```
Tarayıcınızda açılan arayüz üzerinden mevzuatla ilgili sorular sorabilir, kaynak maddeleri inceleyebilir ve token maliyetini anlık takip edebilirsiniz.

---

## 🔧 Pipeline Adımlarını Baştan Çalıştırma (Opsiyonel)

1. **PDF OCR:** `python run_ocr_pages.py`
2. **Metin Birleştirme & Bölümleme:** `python merge_and_split.py`
3. **Teable Senkronizasyonu:** `python sync_to_teable.py --sync-api`

---

## 📜 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında açık kaynak olarak sunulmaktadır.

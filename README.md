# 🏛️ DeepSeek-OCR ile Mevzuat RAG & Yapılandırılmış Bilgi Tabanı Pipeline'ı

> Karmaşık mevzuat ve hukuki belgeleri (**KYK Yurt Hizmetleri Yönetmeliği**) görme-dil modeli (**DeepSeek-OCR**) ile sayfa yapısını koruyarak ayrıştıran, semantik bölümlere ayıran ve **Teable** üzerinde yapılandırılmış bir RAG bilgi tabanına dönüştüren uçtan uca veri işleme pipeline'ı.

---

## 📌 Projeye Genel Bakış

Geleneksel metin çıkarıcılar (PyPDF, pdfplumber, Tesseract OCR vb.), mevzuat metinlerindeki hiyerarşik başlıkları, dipnotları, çok sütunlu düzenleri ve tabloları okurken genellikle yapısal kayıplara yol açar; cümleler sayfa sınırlarında kopar ve anlamsal bütünlük bozulur.

Bu projede **Gelişmiş Görme Tabanlı RAG (Vision-RAG Document Pipeline)** yaklaşımı uygulanmıştır:
1. **Vision-to-Markdown OCR**: **DeepSeek-OCR** (4-bit BitsAndBytes kuantizasyonu) kullanılarak PDF sayfaları doğrudan yapılandırılmış Markdown ve HTML formatına dönüştürülür; başlık, madde ve tablo hiyerarşisi korunur.
2. **Görsel Konumlandırma & Bounding Box**: Metin içi koordinatlar (`<|ref|>...<|det|>`) taranarak sayfa üzerindeki başlık, tablo ve görseller tespit edilir, sınır kutuları çizilir (`_boxed.png`) ve gömülü figürler kırpılır.
3. **Sayfa Sınırlarında Akıllı Satır Onarımı**: Sayfa geçişlerinde bölünen cümleler ve paragraflar anlamsal ve noktalama kurallarına göre onarılıp birleştirilir.
4. **Mevzuat Odaklı Semantik Bölümleme (Semantic Chunking)**: Doküman sabit karakter/token sınırları yerine yönetmeliğin kendi doğal bölümlerine (Bölüm 1-7 ve Kapak) göre mantıksal chunk'lara ayrılır.
5. **Teable Yapılandırılmış Bilgi Tabanı Entegrasyonu**: Elde edilen parçalar; konu başlığı, metadata etiketi, maddeler özeti ve tam markdown içeriği ile birlikte **Teable (Airtable alternatifi / PostgreSQL tabanlı)** veritabanına indekslenir.

---

## 🏗️ Mimari Akış Şeması

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
                  ▼ (Vektör Arama + Metadata Filtreleme + Re-ranking)
             [RAG Uygulaması & LLM Soru-Cevap]
```

---

## 📂 Dizin ve Dosya Yapısı

```text
.
├── 7.5.38510.pdf                  # Kaynak KYK Yurt Hizmetleri Yönetmeliği PDF'i
├── postgresql_tutorial.pdf        # Teknik doküman ve tablo OCR test PDF'i
├── requirements.txt               # Python kütüphane bağımlılıkları
├── .env.example                   # Ortam değişkenleri şablonu (Teable / DB)
├── .gitignore                     # Git tarafından yok sayılacak dosyalar
│
├── run_ocr_pages.py               # DeepSeek-OCR sayfa çıkarma pipeline'ı (4-bit kuantize)
├── process_pdf_pages.py           # vLLM / HuggingFace OCR ve HTML üretici script
├── run_remaining_pages.py         # Kalan sayfalar için toplu OCR çalıştırıcı
├── run_postgresql_ocr.py          # Tablo ve SQL komutları için özelleştirilmiş OCR scripti
├── merge_and_split.py             # Sayfa birleştirme, satır onarma ve semantik bölümleyici
├── sync_to_teable.py              # Bölümleri Teable REST API'ye aktarma ve JSON/CSV export aracı
│
├── teable_exported_data.json      # Teable'a aktarılan 8 semantik bölümün hazır veri seti
├── teable_exported_data.csv       # Bilgi tabanının CSV formatındaki dışa aktarımı
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

Ayrıştırılan mevzuat bölümleri **Teable** tablosunda (`tblbOZHZpc64B44Cuey`) aşağıdaki şema ile saklanır:

| Alan Adı | Veri Tipi | RAG Pipeline'ındaki Rolü |
| :--- | :--- | :--- |
| **`id`** | AutoNumber / Int | Benzersiz bölüm / chunk kimliği |
| **`Subject_String`** | SingleLineText | Bölüm başlığı ve kapsamı (Hızlı arama ve başlık eşleştirme) |
| **`Etiket`** | SingleLineText | Metadata etiketi (Örn: `01_Birinci_Bolum`, `06_Altinci_Bolum`) |
| **`summary`** | LongText | İlgili bölümün kapsadığı maddelerin özeti (Retrieval aşamasında hafif bağlam) |
| **`markdown_content`** | LongText | OCR ile üretilmiş tam Markdown metni (LLM'e verilecek asıl Context) |

---

## 📑 Semantik Bölümleme Listesi (Knowledge Chunks)

Pipeline tarafından üretilen 8 ana bölüm:

1. **`00_Kapak`**: Yönetmelik Başlığı ve Giriş
2. **`01_Birinci_Bolum`**: Amaç, Kapsam, Dayanak ve Tanımlar (Madde 1-4)
3. **`02_Ikinci_Bolum`**: Yurt İhtiyacı, Barınma Şartları, Değerlendirme ve Yerleştirme İşlemleri (Madde 5-10)
4. **`03_Ucuncu_Bolum`**: Yurt Ücreti ve Güvence Bedeli (Madde 11-14)
5. **`04_Dorduncu_Bolum`**: Yurtların Açılması, Kapatılması ve Faaliyet Zamanları (Madde 15-16)
6. **`05_Besinci_Bolum`**: Barınma Süreleri ve İzinler (Madde 17-20)
7. **`06_Altinci_Bolum`**: Disiplin İşlemleri (Madde 21-34)
8. **`07_Yedinci_Bolum`**: Çeşitli ve Son Hükümler (Madde 35-47, Ek Madde 1, Geçici Maddeler)

---

## 🚀 Kurulum ve Kullanım

### 1. Gereksinimler
- Python 3.10 veya üzeri
- CUDA destekli NVIDIA Ekran Kartı (Önerilen: 8GB+ VRAM)
- (Opsiyonel) Yerel Teable kurulumu için Docker

### 2. Kurulum

```bash
# Repoyu klonlayın
git clone https://github.com/yavuz2525/kyk-mevzuat-rag-pipeline.git
cd kyk-mevzuat-rag-pipeline

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 3. OCR Pipeline'ını Çalıştırma

PDF sayfalarını DeepSeek-OCR ile işlemek için:
```bash
python run_ocr_pages.py
```
Bu işlem her sayfa için `output/page_X/` klasöründe:
- `page_X.md`: Markdown çıktısı
- `page_X.html`: HTML önizlemesi
- `page_X_boxed.png`: Tespit edilen alanların kutucuklu görseli
- `page_X_original.png`: Sayfa görseli
- `images/`: Sayfa içinden kırpılan şekil/tabloları üretir.

### 4. Metin Onarma & Semantik Parçalama

Sayfa geçişlerindeki kırık satırları birleştirmek ve mevzuat bölümlerine ayırmak için:
```bash
python merge_and_split.py
```
Çıktılar `output/topics/` dizini altında bölümlere ayrılmış olarak kaydedilir.

### 5. Teable'a Aktarma veya Dışa Aktarma

Verileri `JSON` ve `CSV` formatında dışa aktarmak için:
```bash
python sync_to_teable.py --export-json --export-csv
```

Doğrudan çalışan Teable örneğinize aktarmak için:
```bash
python sync_to_teable.py --sync-api --teable-url "http://localhost:3000" --table-id "<TABLO_ID>" --api-token "<API_TOKEN>"
```

---

## 🔍 RAG ile Örnek Arama & Sorgulama Modeli

```python
import json

# Yapılandırılmış bilgi tabanını yükleyin
with open("teable_exported_data.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

def retrieve_legal_context(user_query: str):
    """
    Kullanıcı sorusuna göre ilgili mevzuat bölümünü getirir.
    (Vektör benzerliği veya metadata filtreleme)
    """
    query_lower = user_query.lower()
    
    if any(k in query_lower for k in ["disiplin", "uyarma", "kınama", "çıkarma", "ceza"]):
        return [c for c in knowledge_base if c["tag"] == "06_Altinci_Bolum"][0]
    elif any(k in query_lower for k in ["ücret", "depozito", "güvence bedeli"]):
        return [c for c in knowledge_base if c["tag"] == "03_Ucuncu_Bolum"][0]
    elif any(k in query_lower for k in ["izin", "süre", "kaç gün", "tatil"]):
        return [c for c in knowledge_base if c["tag"] == "05_Besinci_Bolum"][0]
    
    return knowledge_base[1]

# Örnek sorgu
query = "Yurtta izinsiz dışarıda kalmanın disiplin cezası nedir?"
context = retrieve_legal_context(query)
print(f"Eşleşen Bölüm: {context['subject']}")
print(f"Özet: {context['summary']}")
# context['markdown_content'] LLM prompt'una bağlam olarak eklenir.
```

---

## 📜 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında açık kaynak olarak sunulmaktadır.

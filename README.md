# 🏛️ DeepSeek-OCR ile Mevzuat RAG Pipeline'ı

KYK Yurt Hizmetleri Yönetmeliği'ni **DeepSeek-OCR** ile Markdown'a dönüştüren, mevzuatın doğal bölüm yapısına göre parçalayan, **Teable** ile senkronize edebilen, **OpenAI embeddings + ChromaDB** ile indeksleyen ve **DeepSeek LLM** üzerinden soru-cevap sunan uçtan uca RAG projesi.

## Mimari

```text
7.5.38510.pdf
      │
      ▼
run_ocr_pages.py
      │  DeepSeek-OCR / 4-bit quantization
      ▼
output/page_*/                 # generated, git-ignored
      │
      ▼
merge_and_split.py
      │
      ├── output/merged_all_pages.md
      └── output/topics/       # 8 semantic sections
              │
              ├── sync_to_teable.py ──► Teable
              │
              └── teable_exported_data.json
                              │
                              ▼
                     embed_summaries.py
                              │
                              ▼
                         ChromaDB
                              │
                              ▼
                         rag_chain.py
                              │
                              ▼
                           app.py
```

## Proje Yapısı

```text
.
├── 7.5.38510.pdf
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── .gitignore
│
├── run_ocr_pages.py
├── merge_and_split.py
├── sync_to_teable.py
├── embed_summaries.py
├── query_embeddings.py
├── rag_chain.py
├── app.py
│
├── teable_exported_data.json
└── output/
    ├── merged_all_pages.md
    └── topics/
        ├── 00_Kapak/
        ├── 01_Birinci_Bolum/
        ├── 02_Ikinci_Bolum/
        ├── 03_Ucuncu_Bolum/
        ├── 04_Dorduncu_Bolum/
        ├── 05_Besinci_Bolum/
        ├── 06_Altinci_Bolum/
        └── 07_Yedinci_Bolum/
```

`output/page_*` klasörleri OCR sırasında yeniden oluşturulur ve repoya commit edilmez. Hazır `output/topics/` ile `teable_exported_data.json` ise OCR'ı yeniden çalıştırmadan RAG katmanını kurabilmek için repoda tutulur.

## Kurulum

Python 3.10+ önerilir. OCR adımı için CUDA destekli NVIDIA GPU gerekir; RAG/Streamlit tarafı OCR'dan bağımsız olarak hazır JSON veri setiyle kurulabilir.

```bash
git clone https://github.com/yavuz2525/kyk-mevzuat-rag-pipeline.git
cd kyk-mevzuat-rag-pipeline
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` içinde en az şu anahtarları ayarlayın:

```env
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

Teable kullanacaksanız ayrıca:

```env
TEABLE_URL=http://localhost:3000
TEABLE_API_TOKEN=...
TEABLE_TABLE_ID=...
```

## Hızlı Başlangıç: Hazır Veri Seti ile RAG

OCR'ı yeniden çalıştırmadan:

```bash
python embed_summaries.py
streamlit run app.py
```

Terminalden vektör araması için:

```bash
python query_embeddings.py "öğrencilerin izin hakkı"
```

## Pipeline'ı Baştan Çalıştırma

### 1. OCR

```bash
python run_ocr_pages.py
```

Script PDF'deki sayfa sayısını otomatik algılar ve sayfaları `output/page_N/` altında üretir:

```text
output/page_N/
├── page_N.md
├── page_N.html
├── page_N_original.png
├── page_N_boxed.png
└── images/
```

### 2. Birleştirme ve Semantik Bölümleme

```bash
python merge_and_split.py
```

Bu adım `output/merged_all_pages.md` ve `output/topics/` içeriğini üretir. Script, tüm OCR sayfa girdilerini ve yedi bölüm sınırını doğrulamadan mevcut hazır çıktıları değiştirmez; eksik OCR sonucu varsa güvenli biçimde hata verir.

### 3. Yerel JSON/CSV export

JSON varsayılan exporttur:

```bash
python sync_to_teable.py --export-json
```

CSV gerekiyorsa:

```bash
python sync_to_teable.py --export-csv
```

CSV üretilmiş bir artefakt olduğu için `.gitignore` içindedir.

### 4. Teable senkronizasyonu

```bash
python sync_to_teable.py --sync-api
```

Senkronizasyon **idempotent** çalışır: `Etiket` değeri mevcutsa kayıt güncellenir, yoksa oluşturulur. Aynı komutu tekrar çalıştırmak normal koşullarda aynı bölümlerin yeni kopyalarını oluşturmaz. Daha önceden oluşmuş duplicate kayıtlar otomatik silinmez; veri kaybını önlemek için yalnızca uyarı verilir.

## Teable Şeması

| Alan | Tip | Amaç |
| --- | --- | --- |
| `Subject_String` | SingleLineText | Bölüm başlığı |
| `Etiket` | SingleLineText | Bölümün stabil anahtarı (`01_Birinci_Bolum` vb.) |
| `summary` | LongText | Embedding için kısa özet |
| `markdown_content` | LongText | LLM'e verilen tam mevzuat içeriği |

`Etiket` alanının her bölüm için benzersiz tutulması önerilir.

## Maliyet Ayarları

Model fiyatları zaman içinde değişebileceği için `.env` üzerinden değiştirilebilir:

```env
DEEPSEEK_INPUT_PRICE_PER_M=0.14
DEEPSEEK_OUTPUT_PRICE_PER_M=0.28
```

Streamlit'te yaklaşık TL karşılığı göstermek isterseniz kuru manuel sağlayın:

```env
USD_TRY_RATE=40.0
```

Bu değer boş bırakılırsa yalnızca USD maliyeti gösterilir; uygulama güncel kur olduğunu iddia etmez.

## Güvenlik ve Repo Hijyeni

- Gerçek `.env` dosyaları git tarafından yok sayılır.
- `.env.example` yalnızca placeholder değerler içerir.
- ChromaDB ve SQLite cache dosyaları commit edilmez.
- OCR sayfa render'ları, HTML önizlemeleri ve PNG'ler yeniden üretilebilir artefaktlardır ve commit edilmez.
- OCR model çıktısındaki koordinatlar executable `eval()` ile değil güvenli `ast.literal_eval()` ile parse edilir.
- DeepSeek-OCR kaynak kodu vendor olarak repoya kopyalanmaz; model `deepseek-ai/DeepSeek-OCR` üzerinden yüklenir.

> Bu proje hukuki danışmanlık sağlamaz. Üretilen yanıtlar kaynak mevzuat bağlamıyla doğrulanmalıdır.

## Lisans

MIT License. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.

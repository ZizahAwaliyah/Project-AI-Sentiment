from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
import os # Tambahan penting buat Vercel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = FastAPI()

# --- 1. SETUP CORS ---
# Kita izinkan semua asal (allow_origins=["*"]) supaya aman saat deploy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD MODEL (VERSI VERCEL FRIENDLY) ---
# Vercel butuh alamat lengkap (absolute path) supaya bisa nemu file .pkl
try:
    current_dir = os.path.dirname(__file__) # Cari folder tempat main.py berada
    model_path = os.path.join(current_dir, 'model_sentiment_jualan.pkl')
    vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer_jualan.pkl')
    
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)
    print("✅ Model & Vectorizer berhasil dimuat!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    tfidf = None

# --- 3. LOGIKA PEMBERSIH (PREPROCESSING) ---
factory_sw = StopWordRemoverFactory()
stopword_remover = factory_sw.create_stop_word_remover()

def bersihkan_teks(teks):
    teks = str(teks).lower()
    # Hapus huruf berulang (baanget -> banget)
    teks = re.sub(r'(\w)\1{2,}', r'\1', teks) 
    teks = re.sub(r'@[A-Za-z0-9]+', '', teks)
    teks = re.sub(r'http\S+', '', teks)
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)
    teks = stopword_remover.remove(teks)
    return teks

# --- 4. DAFTAR KATA KUNCI (JALUR VIP) ---
KATA_NEGATIF_KUAT = [
    "jelek", "rusak", "hancur", "kecewa", "buruk", "parah", "penipu", 
    "lambat", "lama", "mahal", "palsu", "bohong", "nyesel", "bekas",
    "galau", "sedih", "benci", "marah"
]

KATA_POSITIF_KUAT = [
    "bagus", "mantap", "keren", "puas", "suka", "cinta", "cantik", 
    "cepat", "rapi", "aman", "oke", "ok", "top", "best", "ramah"
]

# --- 5. ENDPOINT UTAMA ---
class RequestText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Halo! Server AI Sentimen Analisis sudah aktif di Vercel 🚀"}

@app.post("/predict")
def predict_sentiment(request: RequestText):
    if model is None:
        return {"error": "Model belum dimuat dengan benar."}

    teks_asli = request.text
    teks_bersih = bersihkan_teks(teks_asli)
    
    # --- LOGIKA HYBRID ---
    
    # Cek Jalur VIP
    cek_negatif = any(kata in teks_bersih.split() for kata in KATA_NEGATIF_KUAT)
    cek_positif = any(kata in teks_bersih.split() for kata in KATA_POSITIF_KUAT)
    
    if cek_negatif:
        prediksi_label = "negatif"
        confidence_score = "100% (Manual Override)"
    elif cek_positif:
        prediksi_label = "positif"
        confidence_score = "100% (Manual Override)"
    else:
        # Tanya AI
        teks_vector = tfidf.transform([teks_bersih])
        prediksi_label = model.predict(teks_vector)[0]
        
        # Cek Keyakinan (Threshold)
        probs = model.predict_proba(teks_vector)[0]
        max_prob = max(probs)
        
        if max_prob < 0.60:
            prediksi_label = "netral"
            confidence_score = f"{max_prob * 100:.1f}% (Ragu-ragu)"
        else:
            confidence_score = f"{max_prob * 100:.1f}%"
    
    return {
        "text_original": teks_asli,
        "text_clean": teks_bersih,
        "prediction": prediksi_label,
        "confidence": confidence_score
    }
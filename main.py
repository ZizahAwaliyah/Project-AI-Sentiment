from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = FastAPI()

# --- SETUP CORS ---
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL ---
try:
    model = joblib.load('model_sentiment_jualan.pkl')
    tfidf = joblib.load('tfidf_vectorizer_jualan.pkl')
    print("✅ Model dimuat!")
except:
    print("❌ Model error")

# --- PREPROCESSING ---
factory_sw = StopWordRemoverFactory()
stopword_remover = factory_sw.create_stop_word_remover()

def bersihkan_teks(teks):
    teks = str(teks).lower()
    teks = re.sub(r'(\w)\1{2,}', r'\1', teks) # Hapus huruf berulang
    teks = re.sub(r'@[A-Za-z0-9]+', '', teks)
    teks = re.sub(r'http\S+', '', teks)
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)
    teks = stopword_remover.remove(teks)
    return teks

# --- DAFTAR KATA KUNCI (MANUAL OVERRIDE) ---
KATA_NEGATIF_KUAT = [
    "jelek", "rusak", "hancur", "kecewa", "buruk", "parah", "penipu", 
    "lambat", "lama", "mahal", "palsu", "bohong", "nyesel", "bekas",
    "galau", "sedih", "benci"
]

KATA_POSITIF_KUAT = [
    "bagus", "mantap", "keren", "puas", "suka", "cinta", "cantik", 
    "cepat", "rapi", "aman", "oke", "ok", "top", "best", "ramah"
]

# --- ENDPOINT ---
class RequestText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(request: RequestText):
    teks_asli = request.text
    teks_bersih = bersihkan_teks(teks_asli)
    
    # LOGIKA HYBRID:
    # 1. Cek Negatif Dulu (Prioritas Utama)
    cek_negatif = any(kata in teks_bersih.split() for kata in KATA_NEGATIF_KUAT)
    
    # 2. Cek Positif (Prioritas Kedua)
    cek_positif = any(kata in teks_bersih.split() for kata in KATA_POSITIF_KUAT)
    
    if cek_negatif:
        prediksi_label = "negatif"
        confidence_score = "100% (Manual Override)"
    
    elif cek_positif:
        prediksi_label = "positif"
        confidence_score = "100% (Manual Override)"
        
    else:
        # 3. Kalau tidak ada kata kunci, baru tanya AI
        teks_vector = tfidf.transform([teks_bersih])
        prediksi_label = model.predict(teks_vector)[0]
        
        probs = model.predict_proba(teks_vector)[0]
        max_prob = max(probs)
        
        # Ambang batas keraguan (Threshold)
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
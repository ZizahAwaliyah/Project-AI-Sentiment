"use client";

import { useState } from "react";

// Kita definisikan dulu bentuk datanya (Tipe Data)
interface AnalysisResult {
  text_original: string;
  text_clean: string;
  prediction: string;
  confidence: string;
}

export default function Home() {
  // 1. STATE dengan Tipe Data
  const [inputText, setInputText] = useState<string>("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // 2. FUNGSI: Menghubungi Pelayan (Python API)
  const analyzeSentiment = async () => {
    if (!inputText) return;
    
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Gagal menghubungi server Python. Pastikan uvicorn jalan!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6 bg-gray-50">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
        
        {/* Header */}
        <div className="bg-blue-600 p-6 text-center">
          <h1 className="text-2xl font-bold text-white">
            🔮 AI Sentiment Analysis
          </h1>
          <p className="text-blue-100 text-sm mt-1">
            Deteksi emosi pelanggan otomatis
          </p>
        </div>

        {/* Body */}
        <div className="p-6 space-y-4">
          
          {/* Input Area */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Masukkan Komentar / Review:
            </label>
            <textarea
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition h-32 text-gray-800"
              placeholder="Contoh: Barangnya bagus banget, pengiriman cepat..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
          </div>

          {/* Tombol Action */}
          <button
            onClick={analyzeSentiment}
            disabled={loading}
            className={`w-full py-3 rounded-lg font-semibold text-white transition-all
              ${loading 
                ? "bg-gray-400 cursor-not-allowed" 
                : "bg-blue-600 hover:bg-blue-700 shadow-md hover:shadow-lg"
              }`}
          >
            {loading ? "Sedang Menganalisis..." : "Cek Sentimen 🚀"}
          </button>

          {/* Hasil Analisis */}
          {result && (
            <div className={`mt-6 p-4 rounded-lg border-l-4 ${
              result.prediction === "positif" 
                ? "bg-green-50 border-green-500 text-green-800" 
                : result.prediction === "negatif"
                ? "bg-red-50 border-red-500 text-red-800"
                : "bg-gray-100 border-gray-500 text-gray-800" // <--- Warna untuk NETRAL
            }`}>
              <h3 className="font-bold text-lg mb-1 flex items-center gap-2">
                Hasil: {result.prediction.toUpperCase()} 
                <span>
                  {result.prediction === "positif" ? "😄" : 
                   result.prediction === "negatif" ? "😡" : "😐"} 
                </span>
              </h3>
              
              <div className="text-sm opacity-80 space-y-1">
                <p>Confidence: <span className="font-mono font-bold">{result.confidence}</span></p>
                <p>Teks Bersih: <span className="italic">"{result.text_clean}"</span></p>
              </div>
            </div>
          )}

        </div>
      </div>
    </main>
  );
}
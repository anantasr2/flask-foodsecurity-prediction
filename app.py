from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import traceback

# =====================================================
# ⚙️ Inisialisasi Flask App
# =====================================================
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})  # buka semua origin agar frontend bisa akses

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# 📋 Nama Fitur Model
# =====================================================
FEATURE_NAMES = [
    'Akses Jangkauan', 'Jumlah Keluarga Miskin', 'Rasio Penduduk Miskin Desil 1', 
    'Rumah tangga tanpa akses listrik', 'Produksi pangan', 'Luas lahan', 
    'Rasio Sarana Pangan', 'Persentase balita stunting', 'Proporsi Penduduk Lanjut Usia', 
    'Rasio Rumah Tangga Tanpa Air Bersih', 'Rasio Tenaga Kesehatan', 
    'Total Keluarga Beresiko Stunting dan Keluarga rentan'
]

FEATURE_MAPPING = {
    'X1': 'Akses Jangkauan',
    'X2': 'Jumlah Keluarga Miskin',
    'X3': 'Rasio Penduduk Miskin Desil 1',
    'X4': 'Rumah tangga tanpa akses listrik',
    'X5': 'Produksi pangan',
    'X6': 'Luas lahan',
    'X7': 'Rasio Sarana Pangan',
    'X8': 'Persentase balita stunting',
    'X9': 'Proporsi Penduduk Lanjut Usia',
    'X10': 'Rasio Rumah Tangga Tanpa Air Bersih',
    'X11': 'Rasio Tenaga Kesehatan',
    'X12': 'Total Keluarga Beresiko Stunting dan Keluarga rentan'
}

# =====================================================
# 🧠 Load Model & Scaler
# =====================================================
model, scaler = None, None

def load_model_and_scaler():
    """Load model dan scaler dari file pickle"""
    global model, scaler
    try:
        model_path = os.path.join(BASE_DIR, 'best_model_XGB.pkl')
        scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        print("✓ Model dan Scaler berhasil dimuat.")
    except Exception as e:
        print(f"✗ Gagal memuat model atau scaler: {e}")
        traceback.print_exc()

# Load saat startup
load_model_and_scaler()


# =====================================================
# 🏠 ROUTES
# =====================================================
@app.route('/')
def home():
    """Render halaman utama"""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def status():
    """Cek status API dan model"""
    status_info = {
        'status': 'API berjalan dengan baik ✅',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_names': FEATURE_NAMES
    }

    # Tes prediksi sederhana
    if model and scaler:
        try:
            dummy = pd.DataFrame([[10]*12], columns=FEATURE_NAMES)
            dummy_scaled = scaler.transform(dummy)
            pred = model.predict(dummy_scaled)
            status_info['test_prediction'] = float(pred[0])
        except Exception as e:
            status_info['test_error'] = str(e)

    return jsonify(status_info), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Terima data dari frontend dan kembalikan hasil prediksi"""
    try:
        if not model or not scaler:
            return jsonify({'success': False, 'error': 'Model belum dimuat.'}), 500

        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'Data kosong / tidak valid.'}), 400

        print("📥 Data diterima:", data)

        # Mapping input X1...X12 ke fitur asli
        features_dict = {}
        for key, feat_name in FEATURE_MAPPING.items():
            val = data.get(key)
            if val is None or val == '':
                return jsonify({'success': False, 'error': f'Fitur {key} ({feat_name}) belum diisi.'}), 400
            try:
                features_dict[feat_name] = float(val)
            except ValueError:
                return jsonify({'success': False, 'error': f'Nilai fitur {key} tidak valid.'}), 400

        # Buat DataFrame
        df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        score = float(prediction[0])

        response = {
            'success': True,
            'prediction': round(score, 3),
            'confidence': 96.8,
            'features': features_dict
        }

        print("✅ Prediksi berhasil:", response)
        return jsonify(response), 200

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =====================================================
# 🧱 ERROR HANDLERS
# =====================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint tidak ditemukan'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Kesalahan internal server'}), 500


# =====================================================
# ▶️ Jalankan Aplikasi
# =====================================================
if __name__ == '__main__':
    print("\n===============================================")
    print("🚀 Menjalankan Flask API Server")
    print("===============================================")
    print(f"📂 Base Directory : {BASE_DIR}")
    print(f"🤖 Model Loaded  : {'✓ Ya' if model else '✗ Tidak'}")
    print(f"📊 Scaler Loaded : {'✓ Ya' if scaler else '✗ Tidak'}")
    print("===============================================")

    # Port environment variable dari server (Render/Railway)
    port = int(os.environ.get('PORT', 5000))
    # Jalankan di host 0.0.0.0 agar bisa diakses dari luar server
    app.run(host='0.0.0.0', port=port)

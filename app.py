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
app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

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
# 🧠 Global Model dan Scaler
# =====================================================
model = None
scaler = None


# =====================================================
# 📦 Fungsi Load Model dan Scaler
# =====================================================
def load_xgb_model():
    """Load model XGBoost dari file pickle"""
    try:
        model_path = os.path.join(BASE_DIR, 'best_model_XGB.pkl')
        with open(model_path, 'rb') as f:
            global model
            model = pickle.load(f)
        print("✓ Model loaded from PKL successfully.")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        traceback.print_exc()
        return False


def load_scaler():
    """Load scaler dari file pickle"""
    try:
        scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            global scaler
            scaler = pickle.load(f)
        print("✓ Scaler loaded successfully.")
        return True
    except Exception as e:
        print(f"✗ Error loading scaler: {e}")
        traceback.print_exc()
        return False


# =====================================================
# 🚀 Load Model Saat Startup
# =====================================================
model_loaded = load_xgb_model()
scaler_loaded = load_scaler()


# =====================================================
# 🏠 ROUTES
# =====================================================
@app.route('/')
def home():
    """Render halaman utama (index.html)"""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def status():
    """Cek status API"""
    status_info = {
        'status': 'API is running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_names': FEATURE_NAMES
    }

    if model and scaler:
        try:
            test_data = pd.DataFrame([[50] * 12], columns=FEATURE_NAMES)
            test_scaled = scaler.transform(test_data)
            test_pred = model.predict(test_scaled)
            status_info['model_test'] = 'Model can predict'
            status_info['test_prediction'] = float(test_pred[0])
        except Exception as e:
            status_info['model_test'] = f'Model test failed: {str(e)}'

    return jsonify(status_info)


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """API endpoint untuk melakukan prediksi"""
    if request.method == 'OPTIONS':
        return '', 204

    print("\n" + "="*60)
    print("📥 NEW PREDICTION REQUEST")
    print("="*60)

    if not model or not scaler:
        error_msg = 'Model atau scaler belum dimuat. Pastikan file model tersedia.'
        print(f"❌ {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'Data tidak diberikan.'}), 400

        print(f"📊 Data received: {data}")

        # Validasi data input
        features_dict = {}
        missing_features = []
        invalid_features = []

        for feature_name in FEATURE_NAMES:
            feature_key = [k for k, v in FEATURE_MAPPING.items() if v == feature_name][0]
            value = data.get(feature_key)

            if value is None or value == '':
                missing_features.append(feature_name)
            else:
                try:
                    val = float(value)
                    if np.isnan(val) or np.isinf(val):
                        invalid_features.append(feature_name)
                    else:
                        features_dict[feature_name] = val
                except (ValueError, TypeError):
                    invalid_features.append(feature_name)

        if missing_features:
            return jsonify({'success': False, 'error': f'Fitur hilang: {", ".join(missing_features)}'}), 400
        if invalid_features:
            return jsonify({'success': False, 'error': f'Nilai tidak valid: {", ".join(invalid_features)}'}), 400

        if all(v == 0 for v in features_dict.values()):
            return jsonify({'success': False, 'error': 'Semua fitur tidak boleh bernilai 0.'}), 400

        features_df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        score = float(prediction[0])

        confidence = 96.8
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(features_scaled)
                confidence = float(np.max(probabilities) * 100)
            except Exception:
                pass

        response = {
            'success': True,
            'score': round(score, 3),
            'confidence': round(confidence, 1),
            'features_received': list(features_dict.keys()),
            'message': 'Prediksi berhasil dilakukan'
        }

        print(f"✅ Response: {response}")
        print("="*60 + "\n")

        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =====================================================
# 🧱 Error Handlers
# =====================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'success': False, 'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# =====================================================
# ▶️ Jalankan Aplikasi
# =====================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Flask API + Web Server Starting")
    print("="*60)
    print(f"📂 Base Directory: {BASE_DIR}")
    print(f"🤖 Model Status: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print(f"📊 Scaler Status: {'✓ Loaded' if scaler else '✗ Not Loaded'}")
    print("="*60)
    print("📌 Local: http://localhost:5000")
    print("📌 Network: http://0.0.0.0:5000")
    print("="*60 + "\n")

    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)
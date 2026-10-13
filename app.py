# Web画面起動 → 画像を受信 → OpenCVで画像変換 → WebPredictorでAI推論 → JSONで判定結果を返す
# → エラー時はエラー内容を返す → FlaskでローカルWebサーバーを起動

# app.py


from flask import Flask, render_template, request, jsonify
import cv2, numpy as np
from predict_web import WebPredictor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
predictor = None
startup_error = None
try:
    predictor = WebPredictor()
except Exception as e:
    startup_error = str(e)

@app.get('/')
def index():
    return render_template('pet_web.html', startup_error=startup_error)

@app.post('/api/predict')
def predict():
    if predictor is None:
        return jsonify({'error': startup_error or 'モデルを初期化できませんでした。'}), 500
    if 'image' not in request.files:
        return jsonify({'error':'画像が送信されていません。'}), 400
    raw = request.files['image'].read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error':'画像を読み込めませんでした。'}), 400
    try:
        return jsonify(predictor.predict(image))
    except Exception as e:
        return jsonify({'error':str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

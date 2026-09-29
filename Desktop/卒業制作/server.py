import os
import io
import base64
import json
import time
import re
from flask import Flask, request, jsonify, render_template_string, make_response
from google import genai
from google.genai import types
from PIL import Image

app = Flask(__name__)

# Gemini APIキー（環境変数から取得）
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# 最新の解析結果を保存する変数
latest_result = {
    "has_data": False,
    "is_bear": False,
    "confidence": 0,
    "description": "まだ画像が受信されていません。",
    "image_b64": ""
}

# HTMLテンプレート
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>熊検知 AI モニター</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; text-align: center; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h1 { margin-bottom: 20px; color: #333; }
        .status-box { padding: 15px; border-radius: 8px; font-size: 24px; font-weight: bold; margin-bottom: 20px; color: white; }
        .danger { background-color: #e74c3c; }
        .safe { background-color: #2ecc71; }
        .waiting { background-color: #95a5a6; }
        .img-container { margin: 20px 0; }
        img { max-width: 100%; height: auto; border-radius: 8px; border: 3px solid #ddd; }
        .details { text-align: left; background: #f8f9fa; padding: 15px; border-radius: 8px; font-size: 16px; line-height: 1.6; }
    </style>

    <script>
        async function fetchStatus() {
            try {
                const response = await fetch('/api/status', {
                    headers: { 'ngrok-skip-browser-warning': 'true' }
                });
                const data = await response.json();
                const contentDiv = document.getElementById('content');

                if (!data.has_data) {
                    contentDiv.innerHTML = `
                        <div class="status-box waiting">画像待機中...</div>
                        <p>ESP32またはテストコマンドからの画像送信を待っています。</p>
                    `;
                } else {
                    const statusClass = data.is_bear ? 'danger' : 'safe';
                    const statusText = data.is_bear ? '⚠️ 危険！熊を検出しました！' : '✅ 安全（熊ではありません）';
                    const confidencePercent = (data.confidence * 100).toFixed(1);

                    contentDiv.innerHTML = `
                        <div class="status-box ${statusClass}">${statusText}</div>
                        <div class="img-container">
                            <img src="data:image/jpeg;base64,${data.image_b64}" alt="撮影画像">
                        </div>
                        <div class="details">
                            <p><strong>判定確率:</strong> ${confidencePercent} %</p>
                            <p><strong>AIによる説明:</strong> ${data.description}</p>
                        </div>
                    `;
                }
            } catch (err) {
                console.error("データ取得エラー:", err);
            }
        }

        setInterval(fetchStatus, 5000);
        window.onload = fetchStatus;
    </script>
</head>
<body>
    <div class="container">
        <h1>🐻 熊検知 AI モニター</h1>
        <div id="content">
            <div class="status-box waiting">読み込み中...</div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    response = make_response(render_template_string(HTML_TEMPLATE))
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

@app.route('/api/status', methods=['GET'])
def get_status():
    response = make_response(jsonify(latest_result))
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

@app.route('/', methods=['POST'])
def receive_image_and_analyze():
    global latest_result

    image_bytes = request.data
    if not image_bytes:
        return jsonify({"error": "No image received"}), 400

    print("画像を受信しました。AIで解析中...")

    try:
        image = Image.open(io.BytesIO(image_bytes))

        # メモリ消費を抑えるため画像サイズを最大800pxにリサイズ
        image.thumbnail((800, 800))

        # 表示用 Base64 文字列を生成
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=80)
        b64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')

        prompt = """
        この画像を分析し、以下の質問に答えてください。
        1. 熊（ツキノワグマ、ヒグマなど）が写っていますか？
        2. 何が写っているか簡単に説明してください。特徴も踏まえて

        回答は必ず以下のJSONフォーマットのみで返してください。
        {"is_bear": true/false, "confidence": 0.0〜1.0, "description": "説明文"}
        """

        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=[image, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                break
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    print(f"503エラー検出。2秒後に再試行します... (試行 {attempt + 1}/3)")
                    time.sleep(2)
                else:
                    raise e

        # 返却テキストからマークダウン記法(```json ... ```)を整形除去
        raw_text = response.text.strip()
        cleaned_text = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_text, flags=re.MULTILINE).strip()

        res_data = json.loads(cleaned_text)

        latest_result = {
            "has_data": True,
            "is_bear": res_data.get("is_bear", False),
            "confidence": res_data.get("confidence", 0),
            "description": res_data.get("description", ""),
            "image_b64": b64_img
        }

        res = make_response(jsonify(res_data), 200)
        res.headers['ngrok-skip-browser-warning'] = 'true'
        return res

    except Exception as e:
        print(f"エラー: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
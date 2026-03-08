from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from datetime import datetime
import tensorflow as tf
from PIL import Image

# ==============================
# 1. Flask Configuration
# ==============================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

model = None
model_error = None
model_path = None
class_names = ["Parasitized", "Uninfected"]


# ==============================
# 2. Model Loading
# ==============================

try:
    for file in os.listdir():
        if file.endswith(".h5"):
            model_path = os.path.join(os.getcwd(), file)
            break

    if model_path:
        print("Loading model:", model_path)

        model = tf.keras.models.load_model(model_path)

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("Model loaded successfully")
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)

    else:
        model_error = "No .h5 model file found"

except Exception as e:
    model_error = str(e)
    print("Model loading failed:", model_error)


# ==============================
# 3. Image Preprocessing
# ==============================

def prepare_image(img_file):

    img = Image.open(img_file)

    if img.mode != "RGB":
        img = img.convert("RGB")

    input_shape = model.input_shape

    if len(input_shape) == 4:
        height = input_shape[1]
        width = input_shape[2]
    else:
        height = 64
        width = 64

    img = img.resize((width, height))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ==============================
# 4 & 5. Prediction Route
# ==============================

@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 500

    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "error": "No file uploaded"
        }), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename"
        }), 400

    try:

        img = prepare_image(file)

        prediction = model.predict(img, verbose=0)

        print("Prediction array:", prediction)

        prob = float(prediction[0][0])

        if prob < 0.5:
            predicted_class = class_names[0]
        else:
            predicted_class = class_names[1]

        parasitized_prob = round((1 - prob) * 100, 2)
        uninfected_prob = round(prob * 100, 2)

        result = {
            "success": True,
            "prediction": predicted_class,
            "confidence": round(max(parasitized_prob, uninfected_prob), 2),
            "probabilities": {
                "Parasitized": parasitized_prob,
                "Uninfected": uninfected_prob
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ==============================
# 6. Home Route
# ==============================

@app.route('/')
def home():

    template_path = os.path.join("templates", "index.html")

    if os.path.exists(template_path):
        return render_template("index.html")

    else:
        return f"""
        <html>
        <head>
        <title>Malaria Detection</title>
        <style>
        body {{
            font-family: Arial;
            background: linear-gradient(135deg,#667eea,#764ba2);
            color:white;
            text-align:center;
            padding:50px;
        }}
        </style>
        </head>

        <body>

        <h1>Malaria Detection Web App</h1>

        <h3>Server running on port 5000</h3>

        <p>Model Status: {"✅ Loaded" if model else "❌ Not Loaded"}</p>

        <p>{model_error if model_error else ""}</p>

        </body>
        </html>
        """


# ==============================
# 7. Health Check
# ==============================

@app.route('/health')
def health():

    return jsonify({
        "server": "running",
        "model_loaded": model is not None
    })


# ==============================
# 8. Application Entry Point
# ==============================

if __name__ == '__main__':

    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    print("\n===========================")
    print("Malaria Detection Server")
    print("===========================")

    if model:
        print("Model Status: Loaded")
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)
    else:
        print("Model Status: Failed")
        print("Error:", model_error)

    print("\nServer starting...")
    print("Open in browser: http://127.0.0.1:5000")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
        use_reloader=False
    )
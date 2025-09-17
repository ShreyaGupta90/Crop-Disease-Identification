from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# -------------------------------
# Load the trained model
# -------------------------------
MODEL_PATH = "best_model.keras"
IMG_SIZE = (256, 256)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Load model once at startup
model = load_model(MODEL_PATH)

app = FastAPI(title="Plant Disease Classifier", version="1.0")

# -------------------------------
# Frontend with Styling
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌱 Krishi Jyoti - Plant Disease Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #d6f5d6; /* Light green tint */
                margin: 0;
                padding: 0;
            }
            header {
                text-align: center;
                padding: 30px 20px;
                background-color: #ffffff;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            header h1 {
                color: #2e7d32;
                margin: 0;
                font-size: 2rem;
            }
            .stats-container {
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                margin: 30px auto;
                max-width: 1200px;
                gap: 20px;
            }
            .stat-box {
                background: white;
                flex: 1 1 calc(25% - 20px);
                min-width: 220px;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.15);
                text-align: center;
                transition: transform 0.2s ease-in-out;
            }
            .stat-box:hover {
                transform: scale(1.05);
            }
            .stat-number {
                font-size: 1.8rem;
                font-weight: bold;
                color: #2e7d32;
            }
            .stat-text {
                font-size: 1rem;
                color: #333;
            }
            .description {
                text-align: center;
                max-width: 800px;
                margin: 10px auto 30px;
                font-size: 1.1rem;
                color: #1b5e20;
            }
            .upload-section {
                background: white;
                max-width: 600px;
                margin: auto;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.15);
                text-align: center;
            }
            button {
                background-color: #2e7d32;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 1rem;
                border-radius: 8px;
                cursor: pointer;
                transition: background 0.2s;
            }
            button:hover {
                background-color: #1b5e20;
            }
            #result {
                margin-top: 20px;
                font-weight: bold;
                font-size: 1.2rem;
                color: #1b5e20;
                background: #e8f5e9;
                padding: 10px;
                border-radius: 8px;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>🌱 Krishi Jyoti - Plant Disease Classifier</h1>
        </header>

        <p class="description">
            🚜 This model is trained with <b>87K RGB images</b> of healthy and diseased crop leaves,
            categorized into <b>38 different classes</b>, achieving <b>97% accuracy</b>.
        </p>

        <div class="stats-container">
            <div class="stat-box">
                <div class="stat-number">87K+</div>
                <div class="stat-text">Images Used for Training</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">38</div>
                <div class="stat-text">Crop Disease Classes</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">97%</div>
                <div class="stat-text">Model Accuracy</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">Real-time</div>
                <div class="stat-text">Prediction Results</div>
            </div>
        </div>

        <div class="upload-section">
            <form id="upload-form">
                <input type="file" id="file-input" accept="image/*" required>
                <br><br>
                <button type="submit">Predict</button>
            </form>
            <div id="result"></div>
        </div>

        <script>
            document.getElementById("upload-form").addEventListener("submit", async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById("file-input");
                if (!fileInput.files.length) return;

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                document.getElementById("result").innerHTML = "⏳ Predicting...";

                const response = await fetch("/predict/", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();
                if (data.predicted_class) {
                    document.getElementById("result").innerHTML =
                        "🌿 Predicted Class: <b>" + data.predicted_class + "</b><br>✅ Confidence: " + (data.confidence * 100).toFixed(2) + "%";
                } else {
                    document.getElementById("result").innerHTML = "❌ Error: " + data.error;
                }
            });
        </script>
    </body>
    </html>
    """

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx])

        return {"predicted_class": predicted_class, "confidence": round(confidence, 4)}
    except Exception as e:
        return {"error": str(e)}

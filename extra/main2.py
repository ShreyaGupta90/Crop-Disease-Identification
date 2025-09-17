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
# Frontend with Styling + Image Preview
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
                background-color: #f0fdf4; /* Softer green */
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background: white;
                border-radius: 20px;
                padding: 40px;
                max-width: 900px;
                width: 90%;
                box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.2);
                text-align: center;
            }
            h1 {
                color: #2e7d32;
                margin-bottom: 15px;
                font-size: 2.5rem;
            }
            p {
                color: #333;
                font-size: 1.1rem;
                margin-bottom: 25px;
                line-height: 1.6;
            }
            input[type="file"] {
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 8px;
                margin-bottom: 15px;
            }
            button {
                background-color: #2e7d32;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                cursor: pointer;
                font-size: 1rem;
                transition: background 0.3s ease;
            }
            button:hover {
                background-color: #1b5e20;
            }
            #preview {
                margin-top: 20px;
                max-width: 300px;
                border-radius: 15px;
                display: none;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            }
            #result {
                margin-top: 20px;
                font-weight: bold;
                font-size: 1.3rem;
                color: #1b5e20;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌱 Krishi Jyoti</h1>
            <p>🚜 This model is trained with <b>87K RGB images</b> of healthy and diseased crop leaves, 
            categorized into <b>38 different classes</b>, achieving <b>97% accuracy</b>.</p>

            <form id="upload-form">
                <input type="file" id="file-input" accept="image/*" required><br>
                <img id="preview" alt="Uploaded Leaf Preview">
                <br>
                <button type="submit">🔍 Predict</button>
            </form>

            <div id="result"></div>
        </div>

        <script>
            const fileInput = document.getElementById("file-input");
            const preview = document.getElementById("preview");

            fileInput.addEventListener("change", () => {
                const file = fileInput.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = "block";
                    }
                    reader.readAsDataURL(file);
                }
            });

            document.getElementById("upload-form").addEventListener("submit", async (e) => {
                e.preventDefault();
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
                        "🌿 <b>" + data.predicted_class + "</b><br>✅ Confidence: " + (data.confidence * 100).toFixed(2) + "%";
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

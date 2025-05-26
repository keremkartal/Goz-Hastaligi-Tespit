import io
import cv2 # OpenCV
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Göz Hastalığı Tespit API")

MODEL_PATH = "tunnig_model.h5"
model = None
CLASS_NAMES = {0: 'cataract', 1: 'glaucoma', 2: 'diabetic_retinopathy', 3: 'normal'}

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        hp_learning_rate = 0.01 
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(f"Model {MODEL_PATH} başarıyla yüklendi ve derlendi.")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        model = None

@app.on_event("startup")
async def startup_event():
    load_model()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

def preprocess_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (200, 200))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_eye_disease(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model yüklenemedi veya mevcut değil."}
    if not file.content_type.startswith("image/"):
        return {"error": "Yüklenen dosya bir resim değil."}

    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = CLASS_NAMES.get(predicted_class_index, "Bilinmeyen Sınıf")
        confidence = float(prediction[0][predicted_class_index])
        
        return {
            "predicted_disease": predicted_class_name,
            "confidence": f"{confidence:.4f}"
        }
    except Exception as e:
        return {"error": f"Tahmin sırasında bir hata oluştu: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
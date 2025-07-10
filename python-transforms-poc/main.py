from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import io

app = FastAPI()

# Run command http://127.0.0.1:8000/classify/
# Load model and feature extractor
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name)
extractor = ViTFeatureExtractor.from_pretrained(model_name)


@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    # Simulated binary classification for PoC
    confidence = probs.max().item()
    is_damaged = confidence > 0.5  # Replace with actual label mapping once trained

    return JSONResponse({
        "damage": is_damaged,
        "confidence": round(confidence, 3)
    })

@app.get("/")
def health_check():
    return {"status": "ok"}


@app.get("/ping")
async def ping():
    return {"msg": "pong"}

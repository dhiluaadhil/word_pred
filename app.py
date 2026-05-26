"""
app.py — Runs on Render's free tier.

No TensorFlow. Only onnxruntime (~50MB) is used for inference.
Expects model.onnx and tokenizer.pkl to be committed to the repo.
"""

import pickle
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Load artifacts at startup ────────────────────────────────────

try:
    session = rt.InferenceSession("model.onnx")
    input_name = session.get_inputs()[0].name
    print("ONNX model loaded.")
except Exception as e:
    raise RuntimeError(f"Could not load model.onnx: {e}")

try:
    with open("tokenizer.pkl", "rb") as f:
        saved = pickle.load(f)
    tokenizer = saved["tokenizer"]
    max_len = saved["max_len"]
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer.word_index)}, max_len: {max_len}")
except Exception as e:
    raise RuntimeError(f"Could not load tokenizer.pkl: {e}")


# ── App ──────────────────────────────────────────────────────────

app = FastAPI(title="Next Word Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    seed_text: str


@app.get("/")
def root():
    return {
        "status": "ready",
        "vocab_size": len(tokenizer.word_index),
        "endpoint": "POST /predict  →  { seed_text: '...' }",
    }


@app.post("/predict")
def predict(req: PredictRequest):
    tokens = tokenizer.texts_to_sequences([req.seed_text.lower()])[0]

    if not tokens:
        raise HTTPException(
            status_code=400,
            detail="None of those words were in the training vocabulary.",
        )

    tokens = pad_sequences([tokens], maxlen=max_len - 1, padding="pre").astype(np.float32)
    pred = session.run(None, {input_name: tokens})[0]
    predicted_index = int(np.argmax(pred))

    next_word = next(
        (w for w, i in tokenizer.word_index.items() if i == predicted_index),
        "unknown",
    )

    return {"seed": req.seed_text, "next_word": next_word}

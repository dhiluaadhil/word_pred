import numpy as np
import pickle
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="Next Word Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state (persists for the lifetime of this process)
state = {
    "model": None,
    "tokenizer": None,
    "max_len": None,
}

MODEL_PATH = "saved_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN_PATH = "max_len.txt"


# ── Request schemas ──────────────────────────────────────────────

class TrainRequest(BaseModel):
    text: str
    epochs: int = 100


class PredictRequest(BaseModel):
    seed_text: str


# ── Helper: build & train ────────────────────────────────────────

def train_model(text: str, epochs: int):
    text = text.lower()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in text.split("."):
        if not line.strip():
            continue
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            seq = token_list[: i + 1]
            input_sequences.append(seq)

    if not input_sequences:
        raise ValueError("Not enough text to generate sequences. Try a longer passage.")

    max_len = max(len(x) for x in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding="pre")

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]

    model = Sequential(
        [
            Embedding(input_dim=total_words, output_dim=32, input_length=max_len - 1),
            LSTM(64),
            Dense(total_words, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(X, y, epochs=epochs, verbose=0)

    # Persist to disk so the model survives a Render deploy (with a persistent disk)
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(MAX_LEN_PATH, "w") as f:
        f.write(str(max_len))

    return model, tokenizer, max_len


def load_saved_state():
    """Load previously trained model from disk, if it exists."""
    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(TOKENIZER_PATH)
        and os.path.exists(MAX_LEN_PATH)
    ):
        state["model"] = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            state["tokenizer"] = pickle.load(f)
        with open(MAX_LEN_PATH) as f:
            state["max_len"] = int(f.read())
        print("Loaded saved model from disk.")


# ── Startup ──────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    load_saved_state()


# ── Routes ───────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "model_ready": state["model"] is not None,
        "endpoints": {
            "POST /train": "Train on new text",
            "POST /predict": "Predict next word",
        },
    }


@app.post("/train")
def train(req: TrainRequest):
    if len(req.text.split()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least a few sentences of text.",
        )
    try:
        model, tokenizer, max_len = train_model(req.text, req.epochs)
        state["model"] = model
        state["tokenizer"] = tokenizer
        state["max_len"] = max_len
        return {
            "status": "trained",
            "vocab_size": len(tokenizer.word_index),
            "epochs": req.epochs,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    if state["model"] is None:
        raise HTTPException(
            status_code=400,
            detail="No model trained yet. POST to /train first.",
        )

    tokenizer = state["tokenizer"]
    model = state["model"]
    max_len = state["max_len"]

    token_list = tokenizer.texts_to_sequences([req.seed_text.lower()])[0]
    if not token_list:
        raise HTTPException(
            status_code=400,
            detail="None of the words in your input were seen during training.",
        )

    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_index = int(np.argmax(predicted))

    next_word = next(
        (w for w, i in tokenizer.word_index.items() if i == predicted_index),
        "unknown",
    )

    return {"seed": req.seed_text, "next_word": next_word}

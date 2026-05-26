"""
train.py — Run this LOCALLY on your machine.

It trains the LSTM model and exports two files:
  - model.onnx      (the trained model, framework-independent)
  - tokenizer.pkl   (the tokenizer + max_len, needed for inference)

Then commit both files to your repo and deploy to Render.

Requirements (local only, NOT needed on the server):
  pip install tensorflow tf2onnx
"""

import pickle
import numpy as np

import tf2onnx
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ── 1. Get text ──────────────────────────────────────────────────

print("Paste or type your training text below.")
print("When done, press Enter twice.\n")

lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

text = " ".join(lines).lower()

if len(text.split()) < 10:
    print("Error: Please provide at least a few sentences.")
    exit(1)


# ── 2. Tokenize ──────────────────────────────────────────────────

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
print(f"\nVocabulary size: {total_words} words")


# ── 3. Build sequences ───────────────────────────────────────────

input_sequences = []
for sentence in text.split("."):
    if not sentence.strip():
        continue
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[: i + 1])

if not input_sequences:
    print("Error: Not enough text to generate sequences.")
    exit(1)

max_len = max(len(x) for x in input_sequences)
padded = pad_sequences(input_sequences, maxlen=max_len, padding="pre")
X, y = padded[:, :-1], padded[:, -1]

print(f"Training sequences: {len(input_sequences)}, max length: {max_len}")


# ── 4. Build & train model ───────────────────────────────────────

model = Sequential([
    Embedding(input_dim=total_words, output_dim=32, input_length=max_len - 1),
    LSTM(64),
    Dense(total_words, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print("\nTraining model...")
model.fit(X, y, epochs=100, verbose=1)


# ── 5. Export to ONNX ────────────────────────────────────────────

print("\nConverting to ONNX...")
input_signature = [tf.TensorSpec(shape=(None, max_len - 1), dtype=tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("Saved: model.onnx")


# ── 6. Export tokenizer + max_len ────────────────────────────────

with open("tokenizer.pkl", "wb") as f:
    pickle.dump({"tokenizer": tokenizer, "max_len": max_len}, f)
print("Saved: tokenizer.pkl")


# ── 7. Quick sanity check ────────────────────────────────────────

import onnxruntime as rt  # pip install onnxruntime

sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name

seed = text.split()[0]  # test with the first word
tokens = tokenizer.texts_to_sequences([seed])[0]
tokens = pad_sequences([tokens], maxlen=max_len - 1, padding="pre").astype(np.float32)
pred = sess.run(None, {input_name: tokens})[0]
predicted_index = int(np.argmax(pred))
next_word = next((w for w, i in tokenizer.word_index.items() if i == predicted_index), "?")

print(f"\nSanity check — seed: '{seed}' → next word: '{next_word}'")
print("\nAll done! Commit model.onnx and tokenizer.pkl to your repo, then deploy.")

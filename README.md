# LSTM Next-Word Predictor — Render Deployment

## Files
```
├── app.py            # FastAPI web service
├── requirements.txt  # Python dependencies
├── render.yaml       # Render deployment config
└── README.md
```

## Local Testing

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://localhost:8000/docs for the interactive API UI.

## Deploy to Render

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your repo
4. Render will detect `render.yaml` automatically, or set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port 10000`
   - **Instance type:** Starter (512MB+ RAM — free tier is too small for TensorFlow)

## API Usage

### Train
```bash
curl -X POST https://your-app.onrender.com/train \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The cat sat on the mat. The cat ate the rat. The rat ran away.",
    "epochs": 100
  }'
```

### Predict
```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "the cat"}'
```

Response:
```json
{"seed": "the cat", "next_word": "sat"}
```

## Notes

- The model lives **in memory**. It resets if the service restarts.
- To persist it across deploys, uncomment the `disk` section in `render.yaml`.
- Use `tensorflow-cpu` (already set) — Render has no GPUs, and the CPU
  build is ~300MB smaller.

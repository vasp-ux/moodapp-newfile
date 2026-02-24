import base64
import csv
import json
import logging
import os
import sys
from datetime import datetime, timedelta

import cv2
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# New improved models from mood-classification-main
NEW_TEXT_DIR = os.path.join(PROJECT_ROOT, "mood-classification-main", "text")
NEW_VISUAL_DIR = os.path.join(PROJECT_ROOT, "mood-classification-main", "visual")
# Legacy dirs kept as fallback references
TEXT_DIR = os.path.join(PROJECT_ROOT, "text based")
VISUAL_DIR = os.path.join(PROJECT_ROOT, "visual_based")
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")

# Ensure project packages are importable regardless of launch directory.
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data import session_storage  # noqa: E402

app = Flask(__name__)
CORS(app)

# Text model: use original (8-class: angry/contempt/disgust/fear/happy/neutral/sad/surprise)
# The new mood-classification-main text model uses a different 6-class scheme (joy/love/etc.)
# and cannot be fused with the visual model or QUICK_CHECKIN_MAP below.
TEXT_MODEL_PATH = os.path.join(TEXT_DIR, "text_emotion_model.pkl")
VECTORIZER_PATH = os.path.join(TEXT_DIR, "tfidf_vectorizer.pkl")
# Canonical label order for visual models.
DEFAULT_EMOTIONS = [
    "angry",    # 0
    "disgust",  # 1
    "fear",     # 2
    "happy",    # 3
    "neutral",  # 4
    "sad",      # 5
    "surprise", # 6
    "contempt", # 7
]

QUICK_CHECKIN_MAP = {
    "good": ("happy", 0.75, "Glad to hear that. Keep your momentum today."),
    "okay": ("neutral", 0.65, "Thanks for checking in. A small break can help you reset."),
    "not_great": ("sad", 0.75, "Thanks for sharing. You are not alone, take this day gently."),
}
AI_MODE = os.getenv("AI_MODE", "false").strip().lower() == "true"

text_model = joblib.load(TEXT_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def _load_visual_model():
    env_model = os.getenv("VISUAL_MODEL_PATH", "").strip()
    candidates = []
    if env_model:
        candidates.append(env_model)
    candidates.extend(
        [
            os.path.join(VISUAL_DIR, "emotion_model.keras"),
            os.path.join(NEW_VISUAL_DIR, "emotion_model.keras"),
        ]
    )

    errors = []
    for model_path in candidates:
        if not os.path.exists(model_path):
            errors.append(f"{model_path} (missing)")
            continue
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            return model, model_path
        except Exception as model_error:
            errors.append(f"{model_path} ({model_error})")
            continue

    raise RuntimeError(
        "Unable to load any visual model. Attempts:\n- " + "\n- ".join(errors)
    )


def _load_emotion_labels(model_path, expected_classes):
    labels_path = os.path.join(os.path.dirname(model_path), "emotion_labels.json")
    if not os.path.exists(labels_path):
        return DEFAULT_EMOTIONS[:expected_classes]

    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if (
            isinstance(labels, list)
            and len(labels) == expected_classes
            and all(isinstance(item, str) for item in labels)
        ):
            return [item.strip().lower() for item in labels]
        logging.warning(
            "Ignoring invalid emotion_labels.json at %s (expected %s labels).",
            labels_path,
            expected_classes,
        )
    except Exception as labels_error:
        logging.warning("Failed reading emotion labels from %s: %s", labels_path, labels_error)

    return DEFAULT_EMOTIONS[:expected_classes]


visual_model, visual_model_path = _load_visual_model()
visual_input_channels = visual_model.input_shape[-1]
visual_output_classes = int(visual_model.output_shape[-1])
EMOTIONS = _load_emotion_labels(visual_model_path, visual_output_classes)

if len(EMOTIONS) != visual_output_classes:
    EMOTIONS = DEFAULT_EMOTIONS[:visual_output_classes]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _get_text_confidence(text_vector, predicted_emotion):
    if hasattr(text_model, "predict_proba"):
        probs = text_model.predict_proba(text_vector)[0]
        class_index = list(text_model.classes_).index(predicted_emotion)
        return float(probs[class_index])
    return 0.5


def _decode_base64_image(image_data):
    if not image_data:
        return None

    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        binary = base64.b64decode(image_data)
        arr = np.frombuffer(binary, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def _predict_emotion_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return {
            "face_detected": False,
            "emotion": "neutral",
            "confidence": 0.0,
        }

    # Select largest detected face.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray[y : y + h, x : x + w]
    # Normalize contrast to improve robustness under uneven lighting.
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, (48, 48)).astype("float32") / 255.0

    if visual_input_channels == 3:
        face = np.stack([face, face, face], axis=-1)
    else:
        face = np.expand_dims(face, axis=-1)

    face = np.expand_dims(face, axis=0)

    probs = visual_model.predict(face, verbose=0)[0]
    emotion = EMOTIONS[int(np.argmax(probs))]
    confidence = float(np.max(probs))

    return {
        "face_detected": True,
        "emotion": emotion,
        "confidence": round(confidence, 4),
    }


def _soft_weekly_summary(overall_rows):
    if not overall_rows:
        return "No weekly pattern yet. Keep checking in and this space will grow with you."

    supportive = 0
    heavy = 0
    for row in overall_rows:
        mood = str(row.get("overall_mood", "neutral")).lower()
        if mood in ["happy", "neutral", "surprise"]:
            supportive += 1
        elif mood in ["sad", "fear", "angry", "disgust", "contempt", "anxious"]:
            heavy += 1

    if heavy == 0:
        return "This week looked mostly steady and balanced. Keep the same routine that supports you."

    if supportive >= heavy:
        return (
            f"This week had both calm and heavy moments. "
            f"You stayed steady in many check-ins, with around {heavy} heavier moment(s)."
        )

    return (
        f"This week felt heavier in several moments. "
        f"Consider small daily resets and reach out if you need support."
    )


def _build_today_support_prompt(overall, trend, suggestions):
    return f"""
You are a gentle student wellbeing assistant.

Current overall mood: {overall.get('overall_mood', 'neutral')}
Severity: {overall.get('severity', 'low')}
Recent trend: {trend}
Helpful suggestions: {", ".join(suggestions) if suggestions else "Take a short calming break"}

Write:
1) one short supportive message (max 2 sentences)
2) one practical nudge sentence

Use warm, non-technical language.
Do not mention AI, models, probabilities, or diagnosis.
"""


def _build_weekly_reflection_prompt(rows):
    mood_counts = {}
    for row in rows:
        mood = str(row.get("overall_mood", "neutral")).lower()
        mood_counts[mood] = mood_counts.get(mood, 0) + 1

    return f"""
You are a student wellbeing coach.

Last 7-day check-in counts by mood:
{mood_counts}

Write a soft weekly reflection in 2-3 short sentences.
Use human-friendly language and avoid percentages.
No diagnosis, no technical wording.
"""


def _ai_mode_available():
    return AI_MODE and session_storage.is_llm_ready()


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.get("/llm/status")
def llm_status():
    return jsonify(session_storage.get_llm_status()), 200


@app.get("/")
def ui_index():
    return send_from_directory(UI_DIR, "index.html")


@app.post("/checkin/quick")
def quick_checkin():
    payload = request.get_json(silent=True) or {}
    feeling = str(payload.get("feeling", "")).strip().lower()

    if feeling not in QUICK_CHECKIN_MAP:
        return jsonify({"error": "Invalid feeling option"}), 400

    mood, confidence, message = QUICK_CHECKIN_MAP[feeling]
    session_storage.save_text_session(mood, confidence)

    llm_message = ""
    llm_source = "none"
    if _ai_mode_available():
        prompt = f"""You are MoodSense, a warm student wellbeing companion.

A student just did a quick check-in and said they are feeling: {feeling}
Their mood is: {mood}

Write one warm, supportive sentence (max 25 words) that acknowledges how they feel.
Do NOT mention AI, models, or algorithms. Be human and gentle."""
        try:
            llm_message = session_storage.get_llm_response(prompt)
            llm_source = getattr(session_storage, "LAST_LLM_SOURCE", "fallback")
        except Exception:
            llm_message = ""
            llm_source = "error"

    return jsonify(
        {
            "saved": True,
            "message": llm_message if llm_message and llm_source not in ("none", "error") else message,
            "display_mood": "good" if mood == "happy" else ("okay" if mood == "neutral" else "not great"),
            "llm_source": llm_source,
        }
    ), 200


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()

    if len(message) < 1:
        return jsonify({"error": "Message is empty"}), 400

    reply = session_storage.get_chat_response(message)
    llm_source = getattr(session_storage, "LAST_LLM_SOURCE", "fallback")
    llm_error = getattr(session_storage, "LAST_LLM_ERROR", "")

    return jsonify(
        {
            "reply": reply,
            "llm_source": llm_source,
            "llm_error": llm_error,
        }
    ), 200


@app.post("/predict/text")
def predict_text():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if len(text) < 3:
        return jsonify({"error": "Text too short"}), 400

    text_vector = vectorizer.transform([text])
    emotion = text_model.predict(text_vector)[0]
    confidence = round(_get_text_confidence(text_vector, emotion), 4)

    if payload.get("save_session", True):
        session_storage.save_text_session(emotion, confidence)

    # LLM-enhanced analysis
    llm_response = ""
    llm_source = "none"
    if _ai_mode_available():
        prompt = f"""You are a warm, empathetic student wellbeing companion called MoodSense.

A student just shared this reflection about their day:
"{text}"

Our analysis detected their mood as: {emotion} (confidence: {confidence})

Please respond with:
1. A brief, warm acknowledgment of what they shared (1-2 sentences)
2. A gentle insight about their emotional state (1 sentence)
3. One small, practical self-care suggestion (1 sentence)

Rules:
- Be warm, human, and conversational
- Do NOT mention AI, models, algorithms, or confidence scores
- Do NOT diagnose or give medical advice
- Keep your total response under 80 words
- Use a supportive, non-judgmental tone"""
        try:
            llm_response = session_storage.get_llm_response(prompt)
            llm_source = getattr(session_storage, "LAST_LLM_SOURCE", "fallback")
        except Exception:
            llm_response = ""
            llm_source = "error"

    companion_synced = False
    if llm_response:
        try:
            session_storage.sync_reflection_to_companion(text, llm_response)
            companion_synced = True
        except Exception:
            companion_synced = False

    return jsonify(
        {
            "emotion": emotion,
            "confidence": confidence,
            "saved": bool(payload.get("save_session", True)),
            "llm_response": llm_response,
            "llm_source": llm_source,
            "companion_reply": llm_response,
            "companion_source": llm_source,
            "companion_synced": companion_synced,
        }
    ), 200


@app.post("/visual/predict-frame")
def visual_predict_frame():
    payload = request.get_json(silent=True) or {}
    frame = _decode_base64_image(payload.get("image", ""))

    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    result = _predict_emotion_from_frame(frame)
    return jsonify(result), 200


@app.post("/visual/save-session")
def visual_save_session():
    payload = request.get_json(silent=True) or {}
    mood = str(payload.get("mood", "neutral")).strip().lower()

    try:
        confidence = float(payload.get("confidence", 0.5))
    except Exception:
        confidence = 0.5

    confidence = max(0.0, min(confidence, 1.0))

    if mood not in EMOTIONS:
        mood = "neutral"

    session_id = session_storage.save_visual_session(mood, confidence)

    return jsonify(
        {
            "saved": True,
            "session_id": session_id,
            "message": "Energy video check saved.",
        }
    ), 200


@app.post("/fuse")
def fuse_latest():
    payload = request.get_json(silent=True) or {}

    text_result = payload.get("text_result")
    if text_result:
        session_storage.save_text_session(
            text_result.get("mood", "neutral"),
            float(text_result.get("confidence", 0.5)),
        )

    visual_result = payload.get("visual_result")
    if visual_result:
        session_storage.save_visual_session(
            visual_result.get("mood", "neutral"),
            float(visual_result.get("confidence", 0.5)),
        )

    overall = session_storage.save_overall_session()
    if not overall:
        return jsonify({"error": "Please complete at least one text and one visual check-in first."}), 400

    trend = session_storage.analyze_trend()
    suggestions = session_storage.generate_suggestions(overall, trend)
    if _ai_mode_available():
        prompt = _build_today_support_prompt(overall, trend, suggestions)
        llm_message = session_storage.get_llm_response(prompt)
    else:
        prompt = session_storage.build_llm_prompt(overall, trend, suggestions)
        llm_message = session_storage.get_llm_response(prompt)

    llm_source = getattr(session_storage, "LAST_LLM_SOURCE", "fallback")
    llm_error = getattr(session_storage, "LAST_LLM_ERROR", "")
    mode = "ai" if llm_source in ["openrouter", "gemini", "ollama"] else "fallback"

    return jsonify(
        {
            "message": llm_message,
            "nudge": suggestions[0] if suggestions else "Take one slow breath and check in with yourself.",
            "mode": mode,
            "llm_source": llm_source,
            "llm_error": llm_error,
        }
    ), 200


@app.get("/insights/weekly")
def weekly_insights():
    rows = []
    cutoff = datetime.now() - timedelta(days=7)

    try:
        with open(session_storage.OVERALL_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get("timestamp", "")
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if dt >= cutoff:
                    rows.append(row)
    except FileNotFoundError:
        rows = []

    if _ai_mode_available():
        prompt = _build_weekly_reflection_prompt(rows)
        summary = session_storage.get_llm_response(prompt)
        llm_source = getattr(session_storage, "LAST_LLM_SOURCE", "fallback")
        llm_error = getattr(session_storage, "LAST_LLM_ERROR", "")
        mode = "ai" if llm_source in ["openrouter", "gemini", "ollama"] else "fallback"
    else:
        summary = _soft_weekly_summary(rows)
        llm_source = "fallback"
        llm_error = "ai mode disabled or provider unavailable"
        mode = "fallback"

    return jsonify(
        {
            "summary": summary,
            "checkins": len(rows),
            "mode": mode,
            "llm_source": llm_source,
            "llm_error": llm_error,
        }
    ), 200


@app.get("/insights/charts")
def chart_insights():
    try:
        days = int(request.args.get("days", 7))
    except (TypeError, ValueError):
        days = 7

    try:
        weeks = int(request.args.get("weeks", 8))
    except (TypeError, ValueError):
        weeks = 8

    days = max(1, min(days, 30))
    weeks = max(1, min(weeks, 26))

    chart_data = session_storage.get_chart_data(days=days, weeks=weeks)

    return jsonify(
        {
            "days": days,
            "weeks": weeks,
            "daily": chart_data.get("daily", []),
            "weekly": chart_data.get("weekly", []),
            "stats": chart_data.get("stats", {}),
            "saved_weekly_file": chart_data.get("saved_weekly_file", ""),
        }
    ), 200


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)

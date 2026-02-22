import csv
import os
from datetime import datetime

from dotenv import load_dotenv
import requests

# ================== ENV LOAD ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

# ================== GEMINI SETUP (REST API) ==================
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
genai = None  # Not using SDK

# ================== OPENROUTER SETUP ==================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "MoodSense").strip()
OPENROUTER_AVAILABLE = bool(OPENROUTER_API_KEY)

# Toggle for demo / fallback
USE_GEMINI = os.getenv("USE_GEMINI", "false").strip().lower() == "true"
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").strip().lower() == "true"
LLM_AVAILABLE = (USE_OPENROUTER and OPENROUTER_AVAILABLE) or (USE_GEMINI and GEMINI_AVAILABLE)
LAST_LLM_SOURCE = "fallback"
LAST_LLM_ERROR = ""

# ================== PATH SETUP ==================

DATA_DIR = CURRENT_DIR
TEXT_FILE = os.path.join(DATA_DIR, "text_sessions.csv")
VISUAL_FILE = os.path.join(DATA_DIR, "visual_sessions.csv")
OVERALL_FILE = os.path.join(DATA_DIR, "overall_sessions.csv")


# ================== INITIALIZATION ==================

def initialize_storage():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["session_id", "mood", "confidence", "timestamp"]
            )

    if not os.path.exists(VISUAL_FILE):
        with open(VISUAL_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["session_id", "mood", "confidence", "timestamp"]
            )

    if not os.path.exists(OVERALL_FILE):
        with open(OVERALL_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["session_id", "overall_mood", "overall_confidence", "severity", "timestamp"]
            )


# ================== SESSION ID HANDLER ==================

def get_next_session_id(file_path):
    try:
        with open(file_path, "r") as f:
            return len(list(csv.reader(f)))
    except FileNotFoundError:
        return 1


# ================== SAVE TEXT SESSION ==================

def save_text_session(mood, confidence):
    initialize_storage()
    session_id = get_next_session_id(TEXT_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(TEXT_FILE, "a", newline="") as f:
        csv.writer(f).writerow([session_id, mood, confidence, timestamp])

    return session_id


# ================== SAVE VISUAL SESSION ==================

def save_visual_session(mood, confidence):
    initialize_storage()
    session_id = get_next_session_id(VISUAL_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(VISUAL_FILE, "a", newline="") as f:
        csv.writer(f).writerow([session_id, mood, confidence, timestamp])

    return session_id


# ================== READ LATEST SESSION ==================

def get_latest_session(file_path):
    with open(file_path, "r") as f:
        rows = list(csv.DictReader(f))
        return rows[-1] if rows else None


# ================== FUSION LOGIC ==================

def fuse_emotions(text_result, visual_result):
    text_conf = float(text_result["confidence"])
    visual_conf = float(visual_result["confidence"])

    overall_mood = (
        text_result["mood"]
        if text_conf >= visual_conf
        else visual_result["mood"]
    )

    overall_confidence = round((text_conf + visual_conf) / 2, 2)
    return overall_mood, overall_confidence


# ================== SEVERITY LOGIC ==================

def determine_severity(mood, confidence):
    if mood in ["sad", "anxious"] and confidence >= 0.8:
        return "high"
    if mood in ["sad", "anxious"] and confidence >= 0.6:
        return "medium"
    return "low"


# ================== SAVE OVERALL SESSION ==================

def save_overall_session():
    initialize_storage()

    text_result = get_latest_session(TEXT_FILE)
    visual_result = get_latest_session(VISUAL_FILE)

    if not text_result or not visual_result:
        return None

    overall_mood, overall_confidence = fuse_emotions(text_result, visual_result)
    severity = determine_severity(overall_mood, overall_confidence)

    session_id = get_next_session_id(OVERALL_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(OVERALL_FILE, "a", newline="") as f:
        csv.writer(f).writerow(
            [session_id, overall_mood, overall_confidence, severity, timestamp]
        )

    return {
        "session_id": session_id,
        "overall_mood": overall_mood,
        "overall_confidence": overall_confidence,
        "severity": severity,
    }


# ================== TREND ANALYSIS ==================

def get_last_n_overall_sessions(n=3):
    with open(OVERALL_FILE, "r") as f:
        rows = list(csv.DictReader(f))
        return rows[-n:] if len(rows) >= n else rows


def analyze_trend(n=3):
    sessions = get_last_n_overall_sessions(n)
    if not sessions:
        return "neutral"

    negative = sum(
        1 for s in sessions if s["overall_mood"] in ["sad", "anxious", "angry"]
    )
    positive = sum(
        1 for s in sessions if s["overall_mood"] == "happy"
    )

    if negative >= 2:
        return "negative"
    if positive >= 2:
        return "positive"
    return "neutral"


# ================== SUGGESTION ENGINE ==================

def get_support_actions(mood):
    return {
        "sad": [
            "Listen to calming music",
            "Try deep breathing",
            "Write your thoughts in a journal",
        ],
        "anxious": [
            "Practice 4-7-8 breathing",
            "Grounding exercise",
            "Listen to soft music",
        ],
        "angry": [
            "Take a short walk",
            "Muscle relaxation",
            "Listen to slow music",
        ],
        "neutral": [
            "Light mindfulness",
            "Relaxing background music",
        ],
        "happy": [
            "Gratitude journaling",
            "Share positivity with a friend",
        ],
    }.get(mood, [])


def generate_suggestions(overall, trend):
    suggestions = []
    mood = overall["overall_mood"]
    severity = overall["severity"]

    suggestions.extend(get_support_actions(mood))

    if severity == "medium":
        suggestions.append("Call a trusted friend or family member")
    if severity == "high":
        suggestions.append("Contact emergency support or helpline")
    if trend == "negative" and severity != "high":
        suggestions.append("Consider talking to someone you trust")

    return list(dict.fromkeys(suggestions))


# ================== LLM PROMPT ==================

def build_llm_prompt(overall, trend, suggestions):
    return f"""
You are a calm, supportive assistant.

Overall mood: {overall['overall_mood']}
Confidence level: {overall['overall_confidence']}
Severity: {overall['severity']}
Recent emotional trend: {trend}

Suggested actions:
{', '.join(suggestions)}

Explain gently and encouragingly.
Do NOT provide medical advice.
"""


# ================== GEMINI / FALLBACK ==================

def _get_openrouter_response(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a calm, supportive wellbeing assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.6,
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        return None

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        # Some providers return segmented content.
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join([p for p in parts if p]).strip() or None

    if isinstance(content, str):
        return content.strip() or None

    return None


def get_llm_response(prompt):
    global LAST_LLM_SOURCE, LAST_LLM_ERROR
    LAST_LLM_SOURCE = "fallback"
    LAST_LLM_ERROR = ""

    if USE_OPENROUTER and OPENROUTER_AVAILABLE:
        try:
            text = _get_openrouter_response(prompt)
            if text:
                LAST_LLM_SOURCE = "openrouter"
                return text
        except Exception as e:
            LAST_LLM_ERROR = f"openrouter: {e}"
            pass

    if USE_GEMINI and GEMINI_AVAILABLE:
        try:
            resp = requests.post(
                GEMINI_URL,
                params={"key": GEMINI_API_KEY},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if text:
                LAST_LLM_SOURCE = "gemini"
                return text
        except Exception as e:
            if LAST_LLM_ERROR:
                LAST_LLM_ERROR = LAST_LLM_ERROR + f" | gemini: {e}"
            else:
                LAST_LLM_ERROR = f"gemini: {e}"
            pass

    if not ((USE_OPENROUTER and OPENROUTER_AVAILABLE) or (USE_GEMINI and GEMINI_AVAILABLE)):
        LAST_LLM_SOURCE = "fallback"
        LAST_LLM_ERROR = "no provider configured"
        return (
            "Based on recent emotional patterns, you seem to be feeling low. "
            "Listening to calming music or practicing breathing exercises may help. "
            "Reaching out to someone you trust can provide emotional support."
        )

    LAST_LLM_SOURCE = "fallback"
    if not LAST_LLM_ERROR:
        LAST_LLM_ERROR = "provider call failed"
    return (
        "You have already taken a positive step by checking in today. "
        "Try one gentle reset, like a slow breathing break, and revisit how you feel after that."
    )


# ================== CHAT COMPANION ==================

chat_history = []


def get_chat_response(user_message):
    """Send a message to the LLM with recent chat context and return its reply."""
    global LAST_LLM_SOURCE, LAST_LLM_ERROR

    chat_history.append({"role": "user", "content": user_message})

    # Build a prompt with the last 5 exchanges for context
    context_lines = []
    for turn in chat_history[-10:]:  # last 5 pairs = 10 items
        prefix = "Student" if turn["role"] == "user" else "MoodSense"
        context_lines.append(f"{prefix}: {turn['content']}")

    prompt = f"""You are MoodSense, a warm and empathetic student wellbeing companion.

Recent conversation:
{chr(10).join(context_lines)}

Rules:
- Be warm, human, and conversational
- Do NOT mention AI, models, algorithms, or confidence scores
- Do NOT diagnose or give medical advice
- Keep your response under 80 words
- Use a supportive, non-judgmental tone
- If a student seems in crisis, gently suggest reaching out to a trusted person or helpline

Respond as MoodSense:"""

    reply = get_llm_response(prompt)
    chat_history.append({"role": "assistant", "content": reply})

    # Keep history manageable (last 20 turns)
    if len(chat_history) > 20:
        del chat_history[:2]

    return reply


def get_llm_status():
    """Return the current LLM provider status."""
    provider = "none"
    model = ""
    available = False

    if USE_OPENROUTER and OPENROUTER_AVAILABLE:
        provider = "openrouter"
        model = OPENROUTER_MODEL
        available = True
    elif USE_GEMINI and GEMINI_AVAILABLE:
        provider = "gemini"
        model = GEMINI_MODEL
        available = True

    return {
        "available": available,
        "provider": provider,
        "model": model,
        "ai_mode": bool(
            (USE_OPENROUTER and OPENROUTER_AVAILABLE)
            or (USE_GEMINI and GEMINI_AVAILABLE)
        ),
    }


# ================== TEST BLOCK ==================

if __name__ == "__main__":
    print("Testing Phase 2 -> 6...")

    save_text_session("sad", 0.72)
    save_visual_session("neutral", 0.65)

    overall = save_overall_session()
    print("Overall session:", overall)

    trend = analyze_trend()
    print("Trend:", trend)

    suggestions = generate_suggestions(overall, trend)
    print("Suggestions:", suggestions)

    prompt = build_llm_prompt(overall, trend, suggestions)
    llm_message = get_llm_response(prompt)

    print("\nLLM Response:\n")
    print(llm_message)

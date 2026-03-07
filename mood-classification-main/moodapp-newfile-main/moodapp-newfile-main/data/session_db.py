import csv
import json
import os
import sqlite3
from datetime import datetime


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.getenv("MOODSENSE_DB_PATH", os.path.join(CURRENT_DIR, "moodsense.db"))
WEEKLY_TARGET = f"{DB_FILE} (table: weekly_overview)"
FACE_PROFILE_TARGET = f"{DB_FILE} (table: face_profiles)"

LEGACY_TEXT_FILE = os.path.join(CURRENT_DIR, "text_sessions.csv")
LEGACY_VISUAL_FILE = os.path.join(CURRENT_DIR, "visual_sessions.csv")
LEGACY_OVERALL_FILE = os.path.join(CURRENT_DIR, "overall_sessions.csv")
LEGACY_WEEKLY_FILE = os.path.join(CURRENT_DIR, "weekly_overview.csv")
LEGACY_RESPONSES_FILE = os.path.join(CURRENT_DIR, "admin_responses.csv")
LEGACY_USERS_FILE = os.path.join(CURRENT_DIR, "users.csv")
LEGACY_ACTIVITY_FILE = os.path.join(CURRENT_DIR, "activity_log.csv")

TEXT_WEIGHT = 0.6
VISUAL_WEIGHT = 0.4

LEGACY_FILE_TO_TABLE = {
    LEGACY_TEXT_FILE.lower(): "text_sessions",
    LEGACY_VISUAL_FILE.lower(): "visual_sessions",
    LEGACY_OVERALL_FILE.lower(): "overall_sessions",
    LEGACY_WEEKLY_FILE.lower(): "weekly_overview",
    LEGACY_RESPONSES_FILE.lower(): "admin_responses",
    LEGACY_USERS_FILE.lower(): "tracked_users",
    LEGACY_ACTIVITY_FILE.lower(): "activity_log",
}

_STORAGE_READY = False


def _connect():
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_mood(mood):
    value = str(mood or "").strip().lower()
    aliases = {
        "joy": "happy",
        "love": "happy",
        "sadness": "sad",
        "anger": "angry",
        "annoyance": "angry",
        "disapproval": "contempt",
        "anxiety": "fear",
        "nervousness": "fear",
    }
    return aliases.get(value, value or "neutral")


normalize_mood = _normalize_mood


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _table_has_rows(conn, table_name):
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    return bool(row and row["count"] > 0)


def _migrate_simple_csv(conn, csv_path, table_name, columns, transform_row):
    if _table_has_rows(conn, table_name) or not os.path.exists(csv_path):
        return

    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return

    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    payload = []
    for row in rows:
        values = transform_row(row)
        if values:
            payload.append(values)

    if payload:
        conn.executemany(sql, payload)


def initialize_storage():
    global _STORAGE_READY
    if _STORAGE_READY and os.path.exists(DB_FILE):
        return

    os.makedirs(CURRENT_DIR, exist_ok=True)

    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS text_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                mood TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visual_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                mood TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS overall_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                overall_mood TEXT NOT NULL,
                overall_confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                input_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                source TEXT NOT NULL,
                meta TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracked_users (
                email TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT '',
                first_login TEXT NOT NULL DEFAULT '',
                last_login TEXT NOT NULL DEFAULT '',
                logins INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                email TEXT NOT NULL DEFAULT '',
                event_type TEXT NOT NULL,
                mood TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                detail TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weekly_overview (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_start TEXT NOT NULL,
                week_end TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                checkins INTEGER NOT NULL DEFAULT 0,
                avg_confidence REAL NOT NULL DEFAULT 0.0,
                dominant_mood TEXT NOT NULL DEFAULT 'none',
                heavy_count INTEGER NOT NULL DEFAULT 0,
                supportive_count INTEGER NOT NULL DEFAULT 0,
                saved_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_profiles (
                email TEXT PRIMARY KEY,
                label TEXT NOT NULL DEFAULT '',
                template_vector TEXT NOT NULL,
                template_hist TEXT NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                match_threshold REAL NOT NULL DEFAULT 0.0,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        _migrate_simple_csv(
            conn,
            LEGACY_TEXT_FILE,
            "text_sessions",
            ["mood", "confidence", "timestamp"],
            lambda row: (
                _normalize_mood(row.get("mood") or row.get("Emotion")),
                max(0.0, min(_safe_float(row.get("confidence", 0.0), 0.0), 1.0)),
                str(row.get("timestamp") or row.get("DateTime") or "").strip(),
            ),
        )
        _migrate_simple_csv(
            conn,
            LEGACY_VISUAL_FILE,
            "visual_sessions",
            ["mood", "confidence", "timestamp"],
            lambda row: (
                _normalize_mood(row.get("mood") or row.get("Emotion")),
                max(0.0, min(_safe_float(row.get("confidence", 0.0), 0.0), 1.0)),
                str(row.get("timestamp") or row.get("DateTime") or "").strip(),
            ),
        )
        _migrate_simple_csv(
            conn,
            LEGACY_OVERALL_FILE,
            "overall_sessions",
            ["overall_mood", "overall_confidence", "severity", "timestamp"],
            lambda row: (
                _normalize_mood(row.get("overall_mood")),
                max(0.0, min(_safe_float(row.get("overall_confidence", 0.0), 0.0), 1.0)),
                str(row.get("severity", "low")).strip().lower() or "low",
                str(row.get("timestamp", "")).strip(),
            ),
        )
        _migrate_simple_csv(
            conn,
            LEGACY_RESPONSES_FILE,
            "admin_responses",
            ["timestamp", "type", "input_text", "response_text", "source", "meta"],
            lambda row: (
                str(row.get("timestamp", "")).strip(),
                str(row.get("type", "")).strip(),
                str(row.get("input") or row.get("input_text") or "").strip(),
                str(row.get("response") or row.get("response_text") or "").strip(),
                str(row.get("source", "")).strip(),
                str(row.get("meta", "")).strip(),
            ),
        )
        _migrate_simple_csv(
            conn,
            LEGACY_USERS_FILE,
            "tracked_users",
            ["email", "name", "provider", "created_at", "first_login", "last_login", "logins"],
            lambda row: (
                str(row.get("email", "")).strip().lower(),
                str(row.get("name", "")).strip(),
                str(row.get("provider", "")).strip(),
                str(row.get("created_at", "")).strip(),
                str(row.get("first_login", "")).strip(),
                str(row.get("last_login", "")).strip(),
                max(0, _safe_int(row.get("logins", 0), 0)),
            ),
        )
        _migrate_simple_csv(
            conn,
            LEGACY_ACTIVITY_FILE,
            "activity_log",
            ["timestamp", "email", "event_type", "mood", "confidence", "detail"],
            lambda row: (
                str(row.get("timestamp", "")).strip(),
                str(row.get("email", "")).strip().lower(),
                str(row.get("event_type", "")).strip(),
                _normalize_mood(row.get("mood", "")),
                max(0.0, min(_safe_float(row.get("confidence", 0.0), 0.0), 1.0)),
                str(row.get("detail", "")).strip(),
            ),
        )
        _migrate_simple_csv(
            conn,
            LEGACY_WEEKLY_FILE,
            "weekly_overview",
            [
                "week_start",
                "week_end",
                "label",
                "checkins",
                "avg_confidence",
                "dominant_mood",
                "heavy_count",
                "supportive_count",
                "saved_at",
            ],
            lambda row: (
                str(row.get("week_start", "")).strip(),
                str(row.get("week_end", "")).strip(),
                str(row.get("label", "")).strip(),
                max(0, _safe_int(row.get("checkins", 0), 0)),
                max(0.0, min(_safe_float(row.get("avg_confidence", 0.0), 0.0), 1.0)),
                _normalize_mood(row.get("dominant_mood", "none")),
                max(0, _safe_int(row.get("heavy_count", 0), 0)),
                max(0, _safe_int(row.get("supportive_count", 0), 0)),
                str(row.get("saved_at", "")).strip(),
            ),
        )

        conn.commit()

    _STORAGE_READY = True


def _read_table_rows(table_name, columns, order_by, limit=None):
    initialize_storage()
    query = f"SELECT {', '.join(columns)} FROM {table_name} ORDER BY {order_by} DESC"
    params = []
    if limit:
        limit = max(1, int(limit))
        query += " LIMIT ?"
        params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    payload = [dict(row) for row in reversed(rows)]
    return payload


def read_legacy_rows(file_path, limit=None):
    key = str(file_path or "").strip().lower()
    table_name = LEGACY_FILE_TO_TABLE.get(key)
    if table_name == "text_sessions":
        return _read_table_rows(table_name, ["session_id", "mood", "confidence", "timestamp"], "session_id", limit=limit)
    if table_name == "visual_sessions":
        return _read_table_rows(table_name, ["session_id", "mood", "confidence", "timestamp"], "session_id", limit=limit)
    if table_name == "overall_sessions":
        return _read_table_rows(
            table_name,
            ["session_id", "overall_mood", "overall_confidence", "severity", "timestamp"],
            "session_id",
            limit=limit,
        )
    if table_name == "admin_responses":
        rows = _read_table_rows(
            table_name,
            ["timestamp", "type", "input_text", "response_text", "source", "meta"],
            "id",
            limit=limit,
        )
        return [
            {
                "timestamp": row.get("timestamp", ""),
                "type": row.get("type", ""),
                "input": row.get("input_text", ""),
                "response": row.get("response_text", ""),
                "source": row.get("source", ""),
                "meta": row.get("meta", ""),
            }
            for row in rows
        ]
    if table_name == "tracked_users":
        return _read_table_rows(
            table_name,
            ["email", "name", "provider", "created_at", "first_login", "last_login", "logins"],
            "last_login",
            limit=limit,
        )
    if table_name == "activity_log":
        return _read_table_rows(
            table_name,
            ["timestamp", "email", "event_type", "mood", "confidence", "detail"],
            "id",
            limit=limit,
        )
    if table_name == "weekly_overview":
        return _read_table_rows(
            table_name,
            [
                "week_start",
                "week_end",
                "label",
                "checkins",
                "avg_confidence",
                "dominant_mood",
                "heavy_count",
                "supportive_count",
                "saved_at",
            ],
            "id",
            limit=limit,
        )
    return []


def log_response(response_type, input_text, response_text, source, meta=None):
    initialize_storage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO admin_responses (timestamp, type, input_text, response_text, source, meta)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                str(response_type or ""),
                str(input_text or ""),
                str(response_text or ""),
                str(source or ""),
                str(meta or ""),
            ),
        )
        conn.commit()


def read_responses(limit=200):
    rows = _read_table_rows(
        "admin_responses",
        ["timestamp", "type", "input_text", "response_text", "source", "meta"],
        "id",
        limit=limit,
    )
    return [
        {
            "timestamp": row.get("timestamp", ""),
            "type": row.get("type", ""),
            "input": row.get("input_text", ""),
            "response": row.get("response_text", ""),
            "source": row.get("source", ""),
            "meta": row.get("meta", ""),
        }
        for row in rows
    ]


def upsert_user(email, name="", provider="", created_at=""):
    initialize_storage()
    email = str(email or "").strip().lower()
    if not email:
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        row = conn.execute(
            "SELECT email, logins, first_login, created_at FROM tracked_users WHERE email = ?",
            (email,),
        ).fetchone()
        if row:
            conn.execute(
                """
                UPDATE tracked_users
                SET name = ?, provider = ?, created_at = ?, last_login = ?, logins = ?
                WHERE email = ?
                """,
                (
                    str(name or ""),
                    str(provider or ""),
                    str(created_at or row["created_at"] or ""),
                    now,
                    max(1, _safe_int(row["logins"], 0) + 1),
                    email,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO tracked_users (email, name, provider, created_at, first_login, last_login, logins)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    email,
                    str(name or ""),
                    str(provider or ""),
                    str(created_at or ""),
                    now,
                    now,
                    1,
                ),
            )
        conn.commit()
    return True


def read_users(limit=500):
    return _read_table_rows(
        "tracked_users",
        ["email", "name", "provider", "created_at", "first_login", "last_login", "logins"],
        "last_login",
        limit=limit,
    )


def _coerce_float_list(values):
    if values is None:
        return []

    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            return []

    if not isinstance(values, (list, tuple)):
        return []

    output = []
    for value in values:
        try:
            output.append(round(float(value), 8))
        except (TypeError, ValueError):
            continue
    return output


def save_face_profile(
    email,
    template_vector,
    template_hist,
    sample_count,
    label="",
    match_threshold=0.0,
    enabled=True,
):
    initialize_storage()
    email = str(email or "").strip().lower()
    vector = _coerce_float_list(template_vector)
    hist = _coerce_float_list(template_hist)
    if not email or not vector or not hist:
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bounded_threshold = max(0.0, min(_safe_float(match_threshold, 0.0), 1.0))
    bounded_count = max(1, _safe_int(sample_count, 1))

    with _connect() as conn:
        row = conn.execute(
            "SELECT created_at FROM face_profiles WHERE email = ?",
            (email,),
        ).fetchone()
        created_at = str(row["created_at"] or now).strip() if row else now
        conn.execute(
            """
            INSERT INTO face_profiles (
                email, label, template_vector, template_hist,
                sample_count, match_threshold, enabled, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                label = excluded.label,
                template_vector = excluded.template_vector,
                template_hist = excluded.template_hist,
                sample_count = excluded.sample_count,
                match_threshold = excluded.match_threshold,
                enabled = excluded.enabled,
                updated_at = excluded.updated_at
            """,
            (
                email,
                str(label or "").strip(),
                json.dumps(vector, separators=(",", ":")),
                json.dumps(hist, separators=(",", ":")),
                bounded_count,
                bounded_threshold,
                1 if enabled else 0,
                created_at,
                now,
            ),
        )
        conn.commit()
    return True


def get_face_profile(email):
    initialize_storage()
    email = str(email or "").strip().lower()
    if not email:
        return None

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT email, label, template_vector, template_hist,
                   sample_count, match_threshold, enabled, created_at, updated_at
            FROM face_profiles
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

    if not row:
        return None

    return {
        "email": str(row["email"] or "").strip().lower(),
        "label": str(row["label"] or "").strip(),
        "template_vector": _coerce_float_list(row["template_vector"]),
        "template_hist": _coerce_float_list(row["template_hist"]),
        "sample_count": max(0, _safe_int(row["sample_count"], 0)),
        "match_threshold": max(0.0, min(_safe_float(row["match_threshold"], 0.0), 1.0)),
        "enabled": bool(int(row["enabled"] or 0)),
        "created_at": str(row["created_at"] or "").strip(),
        "updated_at": str(row["updated_at"] or "").strip(),
    }


def read_face_profiles(limit=500):
    initialize_storage()
    query = """
        SELECT email, label, sample_count, match_threshold, enabled, created_at, updated_at
        FROM face_profiles
        ORDER BY updated_at DESC, email ASC
    """
    params = []
    if limit:
        query += " LIMIT ?"
        params.append(max(1, int(limit)))

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        {
            "email": str(row["email"] or "").strip().lower(),
            "label": str(row["label"] or "").strip(),
            "sample_count": max(0, _safe_int(row["sample_count"], 0)),
            "match_threshold": round(max(0.0, min(_safe_float(row["match_threshold"], 0.0), 1.0)), 4),
            "enabled": bool(int(row["enabled"] or 0)),
            "created_at": str(row["created_at"] or "").strip(),
            "updated_at": str(row["updated_at"] or "").strip(),
        }
        for row in rows
    ]


def set_face_profile_enabled(email, enabled):
    initialize_storage()
    email = str(email or "").strip().lower()
    if not email:
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE face_profiles
            SET enabled = ?, updated_at = ?
            WHERE email = ?
            """,
            (1 if enabled else 0, now, email),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_face_profile(email):
    initialize_storage()
    email = str(email or "").strip().lower()
    if not email:
        return False

    with _connect() as conn:
        cursor = conn.execute("DELETE FROM face_profiles WHERE email = ?", (email,))
        conn.commit()
        return cursor.rowcount > 0


def log_activity(email, event_type, mood="", confidence="", detail=""):
    initialize_storage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO activity_log (timestamp, email, event_type, mood, confidence, detail)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                str(email or "").strip().lower(),
                str(event_type or ""),
                _normalize_mood(mood),
                max(0.0, min(_safe_float(confidence, 0.0), 1.0)),
                str(detail or ""),
            ),
        )
        conn.commit()


def read_activity(limit=500):
    return _read_table_rows(
        "activity_log",
        ["timestamp", "email", "event_type", "mood", "confidence", "detail"],
        "id",
        limit=limit,
    )


def _next_session_id(table_name):
    initialize_storage()
    with _connect() as conn:
        row = conn.execute(f"SELECT COALESCE(MAX(session_id), 0) + 1 AS next_id FROM {table_name}").fetchone()
    return int(row["next_id"]) if row else 1


def get_next_session_id(file_path):
    key = str(file_path or "").strip().lower()
    table_name = LEGACY_FILE_TO_TABLE.get(key)
    if table_name in {"text_sessions", "visual_sessions", "overall_sessions"}:
        return _next_session_id(table_name)
    return 1


def save_text_session(mood, confidence):
    initialize_storage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized = _normalize_mood(mood)
    bounded_confidence = max(0.0, min(_safe_float(confidence, 0.0), 1.0))

    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO text_sessions (mood, confidence, timestamp) VALUES (?, ?, ?)",
            (normalized, bounded_confidence, timestamp),
        )
        conn.commit()
        return int(cursor.lastrowid)


def save_visual_session(mood, confidence):
    initialize_storage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized = _normalize_mood(mood)
    bounded_confidence = max(0.0, min(_safe_float(confidence, 0.0), 1.0))

    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO visual_sessions (mood, confidence, timestamp) VALUES (?, ?, ?)",
            (normalized, bounded_confidence, timestamp),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_latest_session(file_path):
    key = str(file_path or "").strip().lower()
    table_name = LEGACY_FILE_TO_TABLE.get(key)
    if table_name not in {"text_sessions", "visual_sessions", "overall_sessions"}:
        return None

    columns = {
        "text_sessions": ["session_id", "mood", "confidence", "timestamp"],
        "visual_sessions": ["session_id", "mood", "confidence", "timestamp"],
        "overall_sessions": ["session_id", "overall_mood", "overall_confidence", "severity", "timestamp"],
    }[table_name]

    with _connect() as conn:
        row = conn.execute(
            f"SELECT {', '.join(columns)} FROM {table_name} ORDER BY session_id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def _fuse_emotions(text_result, visual_result):
    text_conf = max(0.0, min(_safe_float(text_result.get("confidence", 0.0), 0.0), 1.0))
    visual_conf = max(0.0, min(_safe_float(visual_result.get("confidence", 0.0), 0.0), 1.0))
    text_score = text_conf * TEXT_WEIGHT
    visual_score = visual_conf * VISUAL_WEIGHT
    overall_mood = (
        _normalize_mood(text_result.get("mood"))
        if text_score >= visual_score
        else _normalize_mood(visual_result.get("mood"))
    )
    overall_confidence = round(text_score + visual_score, 2)
    return overall_mood, overall_confidence


def _determine_severity(mood, confidence):
    normalized = _normalize_mood(mood)
    bounded_confidence = max(0.0, min(_safe_float(confidence, 0.0), 1.0))
    if normalized in {"sad", "anxious"} and bounded_confidence >= 0.8:
        return "high"
    if normalized in {"sad", "anxious"} and bounded_confidence >= 0.6:
        return "medium"
    return "low"


def save_overall_session():
    initialize_storage()
    text_result = get_latest_session(LEGACY_TEXT_FILE)
    visual_result = get_latest_session(LEGACY_VISUAL_FILE)
    if not text_result or not visual_result:
        return None

    overall_mood, overall_confidence = _fuse_emotions(text_result, visual_result)
    severity = _determine_severity(overall_mood, overall_confidence)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO overall_sessions (overall_mood, overall_confidence, severity, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (overall_mood, overall_confidence, severity, timestamp),
        )
        conn.commit()
        session_id = int(cursor.lastrowid)

    return {
        "session_id": session_id,
        "overall_mood": overall_mood,
        "overall_confidence": overall_confidence,
        "severity": severity,
    }


def get_last_n_overall_sessions(n=3):
    rows = _read_table_rows(
        "overall_sessions",
        ["session_id", "overall_mood", "overall_confidence", "severity", "timestamp"],
        "session_id",
        limit=n,
    )
    return rows


def get_overall_sessions():
    initialize_storage()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT overall_mood, overall_confidence, severity, timestamp
            FROM overall_sessions
            ORDER BY session_id ASC
            """
        ).fetchall()

    sessions = []
    for row in rows:
        timestamp = str(row["timestamp"] or "").strip()
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        sessions.append(
            {
                "timestamp": dt,
                "mood": _normalize_mood(row["overall_mood"]),
                "confidence": max(0.0, min(_safe_float(row["overall_confidence"], 0.0), 1.0)),
                "severity": str(row["severity"] or "low").strip().lower() or "low",
            }
        )

    return sessions


def save_weekly_snapshot(weekly_points):
    initialize_storage()
    saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute("DELETE FROM weekly_overview")
        conn.executemany(
            """
            INSERT INTO weekly_overview (
                week_start, week_end, label, checkins, avg_confidence,
                dominant_mood, heavy_count, supportive_count, saved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(point.get("week_start", "")),
                    str(point.get("week_end", "")),
                    str(point.get("label", "")),
                    max(0, _safe_int(point.get("count", 0), 0)),
                    max(0.0, min(_safe_float(point.get("avg_confidence", 0.0), 0.0), 1.0)),
                    _normalize_mood(point.get("dominant_mood", "none")),
                    max(0, _safe_int(point.get("heavy_count", 0), 0)),
                    max(0, _safe_int(point.get("supportive_count", 0), 0)),
                    saved_at,
                )
                for point in weekly_points
            ],
        )
        conn.commit()

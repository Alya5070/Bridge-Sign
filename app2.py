from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, flash
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from functools import wraps
from typing import Optional
import os
import sys
import platform
import sqlite3
import subprocess
import re
import uuid
from datetime import datetime
from collections import deque
import tensorflow as tf
import shutil
import time
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # psycopg2 might not be installed in all environments
    psycopg2 = None
    RealDictCursor = None
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from data_collection_utils import (
    allowed_file,
    expand_hand_bbox,
    extract_frames_from_video,
    make_sample_dir,
    safe_word_folder,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DYNAMIC_MODEL_PATH = os.path.join(BASE_DIR, "Model", "dynamic_model.h5")
DYNAMIC_LABELS_PATH = os.path.join(BASE_DIR, "Model", "dynamic_labels.txt")
SPELLING_MODEL_PATH = os.path.join(BASE_DIR, "Model", "Static.h5")
SPELLING_LABELS_PATH = os.path.join(BASE_DIR, "Model", "labels.txt")
WORDS_TABLE_NAME = "msl_words"
MODEL_BACKUP_DIR = os.path.join(BASE_DIR, "Model", "Backup Model")
ALLOWED_MODEL_EXT = {"h5"}
ALLOWED_LABEL_EXT = {"txt"}
PROJECT_VENV_PY = os.path.join(BASE_DIR, ".venv1", "Scripts", "python.exe")
FEATURE_FLAG_DATASET_UPLOAD = "dataset_upload_visible"
FEATURE_FLAG_DEFAULTS = {FEATURE_FLAG_DATASET_UPLOAD: False}
SUPABASE_DB_URL = os.environ.get(
    "SUPABASE_DB_URL",
    "",  # Set in environment when available
)


def _env_flag(name: str, default: bool = False) -> bool:
    """
    Read a boolean-like environment variable safely.
    Treats 1/true/yes/on (case-insensitive) as True; everything else is False.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# Force SQLite-first behavior unless the flag is explicitly enabled.
USE_SUPABASE_ONLY = _env_flag("USE_SUPABASE_ONLY", False)

# Ensure the project virtual environment is used even if invoked via another Python.
if os.path.exists(PROJECT_VENV_PY):
    current_python = os.path.realpath(sys.executable)
    target_python = os.path.realpath(PROJECT_VENV_PY)
    if current_python != target_python and not os.environ.get("MSL_USE_PROJECT_VENV"):
        os.environ["MSL_USE_PROJECT_VENV"] = "1"
        os.execv(target_python, [target_python] + sys.argv)


def _load_model_safe(model_path: str):
    """
    Load a keras model, retrying with a compat InputLayer if batch_shape
    appears in the config (common across TF/Keras version mismatches), and
    mapping DTypePolicy for newer saved models.
    """
    custom_objects = {
        # Newer Keras saves policy objects that older TF doesn't know.
        "DTypePolicy": tf.keras.mixed_precision.Policy,
    }

    def _try_load(extra_custom=None, safe_mode=True):
        objs = {**custom_objects, **(extra_custom or {})}
        # Provide both plain and fully-qualified keys to maximize override hit.
        objs.setdefault("keras.layers.InputLayer", objs.get("InputLayer"))
        return tf.keras.models.load_model(model_path, custom_objects=objs, compile=False, safe_mode=safe_mode)

    # First attempt: normal load with custom dtype policy.
    try:
        return _try_load()
    except Exception as exc_first:
        # If batch_shape or InputLayer config issues, try the compat InputLayer.
        class InputLayerCompat(tf.keras.layers.InputLayer):
            @classmethod
            def from_config(cls, config):
                if "batch_shape" in config and "batch_input_shape" not in config:
                    config["batch_input_shape"] = config.pop("batch_shape")
                return super().from_config(config)

        try:
            return _try_load({"InputLayer": InputLayerCompat})
        except Exception as exc_second:
            # Last resort: disable safe_mode to bypass strict checks.
            try:
                return _try_load({"InputLayer": InputLayerCompat}, safe_mode=False)
            except Exception:
                # Re-raise original for clearer error context.
                raise exc_first

def get_supabase_connection():
    """Return a psycopg2 connection to Supabase, or None if unavailable."""
    if not SUPABASE_DB_URL or psycopg2 is None:
        return None
    try:
        conn = psycopg2.connect(
            SUPABASE_DB_URL,
            cursor_factory=RealDictCursor,
            connect_timeout=5,
            sslmode="require",
        )
        conn.autocommit = False
        return conn
    except Exception:
        return None


def _get_supabase_connection():
    """Backward-compatible alias used by older helper functions."""
    return get_supabase_connection()


def _convert_qmarks_to_pg(sql: str) -> str:
    # Simple conversion of sqlite-style ? placeholders to %s for psycopg2.
    return sql.replace("?", "%s")


class SupabaseCursorProxy:
    def __init__(self, cursor):
        self.cursor = cursor

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()


class SupabaseConnectionProxy:
    """
    Minimal sqlite-like facade over psycopg2 connection so existing code paths
    can keep using conn.execute(...).
    """
    def __init__(self, conn):
        self.conn = conn
        self.row_factory = None  # ignored, for API compatibility

    def execute(self, sql: str, params: tuple | list = ()):
        cur = self.conn.cursor()
        sql_pg = _convert_qmarks_to_pg(sql)
        cur.execute(sql_pg, params or ())
        return SupabaseCursorProxy(cur)

    def commit(self):
        try:
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


def ensure_supabase_extensions(cur) -> None:
    """Ensure required extensions exist on Supabase."""
    cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')


def supabase_log_model_version(
    model_type: str,
    version: str,
    notes: str,
    model_filename: str,
    labels_filename: str,
    backup_path: str,
    uploaded_by: str,
) -> None:
    """Record model metadata to Supabase (best-effort)."""
    conn = get_supabase_connection()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_supabase_extensions(cur)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public.model_versions (
                        id uuid NOT NULL DEFAULT uuid_generate_v4(),
                        model_type text NOT NULL,
                        version text NOT NULL,
                        notes text NOT NULL,
                        model_filename text NOT NULL,
                        labels_filename text NOT NULL,
                        backup_path text,
                        uploaded_by text,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        CONSTRAINT model_versions_pkey PRIMARY KEY (id)
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO public.model_versions
                        (model_type, version, notes, model_filename, labels_filename, backup_path, uploaded_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        model_type,
                        version,
                        notes,
                        model_filename,
                        labels_filename,
                        backup_path,
                        uploaded_by,
                    ),
                )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def supabase_get_latest_version(model_type: str) -> Optional[str]:
    """Fetch latest version for a model type from Supabase."""
    conn = get_supabase_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT version
                FROM public.model_versions
                WHERE model_type = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (model_type,),
            )
            row = cur.fetchone()
            if not row:
                return None
            if isinstance(row, dict):
                return row.get("version")
            return row[0]
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def supabase_upsert_admin(username: str, password_hash: str) -> None:
    """Ensure the admin user exists in Supabase."""
    conn = get_supabase_connection()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_supabase_extensions(cur)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public.app_users (
                        id uuid NOT NULL DEFAULT uuid_generate_v4(),
                        username text NOT NULL UNIQUE,
                        password_hash text,
                        camera_enabled boolean NOT NULL DEFAULT true,
                        camera_index integer NOT NULL DEFAULT 0,
                        practice_time_seconds integer NOT NULL DEFAULT 0,
                        is_admin boolean NOT NULL DEFAULT false,
                        migrated_at timestamptz NOT NULL DEFAULT now(),
                        CONSTRAINT app_users_pkey PRIMARY KEY (id)
                    );
                    """
                )
                cur.execute(
                    "ALTER TABLE public.app_users ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false;"
                )
                cur.execute(
                    """
                    INSERT INTO public.app_users
                        (username, password_hash, camera_enabled, camera_index, practice_time_seconds, is_admin)
                    VALUES (%s, %s, true, 0, 0, true)
                    ON CONFLICT (username) DO UPDATE SET
                        password_hash = EXCLUDED.password_hash,
                        is_admin = true,
                        migrated_at = now();
                    """,
                    (username, password_hash),
                )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def ensure_supabase_extensions(cur) -> None:
    """Ensure required extensions exist on Supabase."""
    cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')


def supabase_log_model_version(
    model_type: str,
    version: str,
    notes: str,
    model_filename: str,
    labels_filename: str,
    backup_path: str,
    uploaded_by: str,
) -> None:
    """Record model metadata to Supabase (best-effort)."""
    conn = get_supabase_connection()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_supabase_extensions(cur)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public.model_versions (
                        id uuid NOT NULL DEFAULT uuid_generate_v4(),
                        model_type text NOT NULL,
                        version text NOT NULL,
                        notes text NOT NULL,
                        model_filename text NOT NULL,
                        labels_filename text NOT NULL,
                        backup_path text,
                        uploaded_by text,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        CONSTRAINT model_versions_pkey PRIMARY KEY (id)
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO public.model_versions
                        (model_type, version, notes, model_filename, labels_filename, backup_path, uploaded_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        model_type,
                        version,
                        notes,
                        model_filename,
                        labels_filename,
                        backup_path,
                        uploaded_by,
                    ),
                )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def supabase_upsert_admin(username: str, password_hash: str) -> None:
    """Ensure the admin user exists in Supabase."""
    conn = get_supabase_connection()
    if not conn:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_supabase_extensions(cur)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public.app_users (
                        id uuid NOT NULL DEFAULT uuid_generate_v4(),
                        username text NOT NULL UNIQUE,
                        password_hash text,
                        camera_enabled boolean NOT NULL DEFAULT true,
                        camera_index integer NOT NULL DEFAULT 0,
                        practice_time_seconds integer NOT NULL DEFAULT 0,
                        is_admin boolean NOT NULL DEFAULT false,
                        migrated_at timestamptz NOT NULL DEFAULT now(),
                        CONSTRAINT app_users_pkey PRIMARY KEY (id)
                    );
                    """
                )
                cur.execute(
                    "ALTER TABLE public.app_users ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false;"
                )
                cur.execute(
                    """
                    INSERT INTO public.app_users
                        (username, password_hash, camera_enabled, camera_index, practice_time_seconds, is_admin)
                    VALUES (%s, %s, true, 0, 0, true)
                    ON CONFLICT (username) DO UPDATE SET
                        password_hash = EXCLUDED.password_hash,
                        is_admin = true,
                        migrated_at = now();
                    """,
                    (username, password_hash),
                )
    finally:
        try:
            conn.close()
        except Exception:
            pass

# Supported Malaysian Sign Language (MSL) words for the dynamic model.
# Keep this list aligned with Model/dynamic_labels.txt and Supabase.
HANDSIGN_WORDS_SEED = [
    {
        "word": "Apa",
        "description": "Question gesture for 'what' - palms slightly up with a small shrug.",
        "category": "Question",
    },
    {
        "word": "Kamu",
        "description": "Pronoun for 'you' - index finger points toward the listener.",
        "category": "Pronoun",
    },
    {
        "word": "Makan",
        "description": "Action for 'eat' - hand mimics bringing food to the mouth.",
        "category": "Action",
        "category": "Action",
    },
]


def normalize_word_token(value: str) -> str:
    """Normalize a predicted/typed value into a comparable token."""
    if not value:
        return ""
    return "".join(ch for ch in str(value).upper() if ch.isalpha())


def _prepare_hand_sign_words() -> list[dict]:
    """Attach canonical tokens to the seed entries."""
    prepared = []
    for entry in HANDSIGN_WORDS_SEED:
        word_text = entry.get("word", "").strip()
        if not word_text:
            continue
        prepared.append({
            "word": word_text.upper(),
            "description": entry.get("description", ""),
            "category": entry.get("category", "Core vocabulary"),
            "canonical": normalize_word_token(word_text),
        })
    return prepared


AVAILABLE_HANDSIGN_WORDS = _prepare_hand_sign_words()
WORD_TOKEN_PATTERN = re.compile(r"[A-Za-z]+")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config["MAX_CONTENT_LENGTH"] = 800 * 1024 * 1024

DB_PATH = os.path.join(BASE_DIR, "instance", "msl_app.db")
BASE_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
BASE_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
HAND_CONTEXT_RATIO = 0.35
HAND_CONTEXT_RATIO_DYNAMIC = HAND_CONTEXT_RATIO
HAND_CONTEXT_RATIO_SPELLING = 0.18
USE_LEGACY_SPELLING_PREPROCESS = True  # Enable legacy spelling crop/pad while keeping model compatibility.
LEGACY_SPELLING_IMG_SIZE = 300  # Legacy canvas size; will be resized down to the model input.
LEGACY_SPELLING_NORMALIZE = True  # Set False to send raw uint8 values to the model.
SPELLING_SMOOTH_WINDOW = 7
SPELLING_CONFIDENCE_THRESHOLD_DEFAULT = 0.45  # Base avg prob threshold before emitting a letter.
SPELLING_CONFIDENCE_THRESHOLD_CONFUSED = 0.6  # Stricter threshold for visually similar letters.
SPELLING_CONFUSED_LABELS = {"V", "W", "U", "G", "Q"}
SPELLING_CONFUSED_GAP = 0.08  # Require margin over runner-up for confused letters.


def init_db() -> None:
    """Ensure the local SQLite database exists with the expected schema."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(80) UNIQUE NOT NULL,
            password_hash VARCHAR(200) NOT NULL,
            camera_enabled BOOLEAN NOT NULL DEFAULT 1,
            camera_index INTEGER NOT NULL DEFAULT 0,
            practice_time_seconds INTEGER NOT NULL DEFAULT 0,
            supabase_user_id TEXT UNIQUE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hand_sign_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            letter TEXT NOT NULL,
            practice_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE(user_id, letter),
            FOREIGN KEY(user_id) REFERENCES user(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            version TEXT NOT NULL,
            notes TEXT NOT NULL,
            model_filename TEXT NOT NULL,
            labels_filename TEXT NOT NULL,
            backup_path TEXT,
            created_by TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_flag (
            key TEXT PRIMARY KEY,
            enabled INTEGER NOT NULL
        )
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(user)").fetchall()}
    if "practice_time_seconds" not in existing_columns:
        conn.execute("ALTER TABLE user ADD COLUMN practice_time_seconds INTEGER NOT NULL DEFAULT 0")
    if "is_admin" not in existing_columns:
        conn.execute("ALTER TABLE user ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0")
    conn.commit()
    conn.close()


def ensure_admin_user(username: str = "Mori", password: str = "1234") -> None:
    """Ensure the admin account exists and is marked as admin locally and in Supabase."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        normalized = username.strip().lower()
        existing = conn.execute(
            "SELECT * FROM user WHERE LOWER(username) = ?",
            (normalized,),
        ).fetchone()
        password_hash = None
        if existing:
            password_hash = existing["password_hash"]
            conn.execute("UPDATE user SET is_admin = 1 WHERE id = ?", (existing["id"],))
        else:
            password_hash = generate_password_hash(password)
            conn.execute(
                """
                INSERT INTO user (username, password_hash, camera_enabled, camera_index, practice_time_seconds, is_admin)
                VALUES (?, ?, 1, 0, 0, 1)
                """,
                (username, password_hash),
            )
        conn.commit()
    finally:
        conn.close()
    supabase_upsert_admin(username, password_hash or "")


init_db()
ensure_admin_user()


def init_feature_flags() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        for key, default in FEATURE_FLAG_DEFAULTS.items():
            row = conn.execute("SELECT enabled FROM feature_flag WHERE key = ?", (key,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO feature_flag (key, enabled) VALUES (?, ?)",
                    (key, 1 if default else 0),
                )
        conn.commit()
    finally:
        conn.close()


def get_feature_flag(key: str, default: bool = False) -> bool:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute("SELECT enabled FROM feature_flag WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return bool(row[0])
    finally:
        conn.close()


def set_feature_flag(key: str, enabled: bool) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO feature_flag (key, enabled) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET enabled=excluded.enabled",
            (key, 1 if enabled else 0),
        )
        conn.commit()
    finally:
        conn.close()


init_feature_flags()


def get_hand_sign_words() -> list[dict]:
    """
    Fetch the list of supported hand-sign words.

    In production this should read from Supabase (see WORDS_TABLE_NAME) and
    reference the assets stored under the Data folder. Until that table exists
    we return a curated, static subset for the learning module.
    """
    return AVAILABLE_HANDSIGN_WORDS


def find_hand_sign_word_entry(normalized_word: str) -> Optional[dict]:
    """Return the configured hand-sign entry that matches the canonical token."""
    if not normalized_word:
        return None
    canonical = normalized_word.upper()
    for entry in get_hand_sign_words():
        if entry["canonical"] == canonical:
            return entry
    return None


def tokenize_phrase_chunks(phrase: str) -> list[tuple[str, str]]:
    """Split an input phrase into (raw, normalized) word tokens."""
    if not phrase:
        return []
    tokens = []
    for chunk in WORD_TOKEN_PATTERN.findall(phrase):
        canonical = normalize_word_token(chunk)
        if canonical:
            tokens.append((chunk, canonical))
    return tokens


def get_db_connection():
    if USE_SUPABASE_ONLY and SUPABASE_DB_URL:
        pg_conn = get_supabase_connection()
        if pg_conn:
            return SupabaseConnectionProxy(pg_conn)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_profile(row: Optional[sqlite3.Row]) -> Optional[dict]:
    if not row:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "camera_enabled": bool(row["camera_enabled"]),
        "camera_index": row["camera_index"],
        "practice_time_seconds": row["practice_time_seconds"],
        "is_admin": bool(row["is_admin"]) if "is_admin" in row.keys() else False,
    }


def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    normalized = normalize_username(username)
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM user WHERE LOWER(username) = ?",
        (normalized,),
    ).fetchone()
    conn.close()
    return user


def get_user_profile(user_id: int) -> Optional[dict]:
    row = get_user_record(user_id)
    return row_to_profile(row)


def get_user_record(user_id: int) -> Optional[sqlite3.Row]:
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM user WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


def username_exists(username: str, exclude_uid: Optional[int] = None) -> bool:
    row = get_user_by_username(username)
    if not row:
        return False
    if exclude_uid is None:
        return True
    return row["id"] != exclude_uid


def create_user(username: str, password: str) -> int:
    hashed = generate_password_hash(password)
    if USE_SUPABASE_ONLY:
        pg_conn = get_supabase_connection()
        if not pg_conn:
            raise RuntimeError("Supabase connection unavailable.")
        try:
            with pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO app_users (username, password_hash, camera_enabled, camera_index, is_admin)
                        VALUES (%s, %s, true, 0, false)
                        ON CONFLICT (username) DO NOTHING
                        RETURNING id
                        """,
                        (username, hashed),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise ValueError("Username already exists.")
                    return row[0] if not isinstance(row, dict) else row.get("id")
        finally:
            try:
                pg_conn.close()
            except Exception:
                pass
    conn = get_db_connection()
    user_id = None
    try:
        cursor = conn.execute(
            "INSERT INTO user (username, password_hash, camera_enabled, camera_index, is_admin) "
            "VALUES (?, ?, 1, 0, 0)",
            (username, hashed),
        )
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except sqlite3.IntegrityError as exc:
        raise ValueError("Username already exists.") from exc
    finally:
        conn.close()
        if user_id is not None:
            sync_supabase_user(user_id)


def verify_user_credentials(username: str, password: str) -> Optional[dict]:
    row = get_user_by_username(username)
    if not row:
        return None
    if not check_password_hash(row["password_hash"], password):
        return None
    return row_to_profile(row)


def update_user_profile(user_id: int, data: dict) -> None:
    if not data:
        return
    columns = []
    values = []
    for key, value in data.items():
        columns.append(f"{key} = ?")
        if key == "camera_enabled":
            values.append(1 if value else 0)
        else:
            values.append(value)
    values.append(user_id)
    conn = get_db_connection()
    conn.execute(f"UPDATE user SET {', '.join(columns)} WHERE id = ?", values)
    conn.commit()
    conn.close()
    sync_supabase_user(user_id, sync_practice_time=False)


def set_supabase_user_id(user_id: int, supabase_user_id: str) -> None:
    if not supabase_user_id:
        return
    conn = get_db_connection()
    conn.execute(
        "UPDATE user SET supabase_user_id = ? WHERE id = ?",
        (supabase_user_id, user_id),
    )
    conn.commit()
    conn.close()


def sync_supabase_user(user_id: int, sync_practice_time: bool = True) -> Optional[str]:
    """
    Ensure the Supabase app_users record exists and matches local settings.

    Returns the Supabase UUID for the user (or None if Supabase is disabled/unreachable).
    """
    if not SUPABASE_DB_URL:
        return None

    if USE_SUPABASE_ONLY:
        # Already operating on Supabase; nothing to sync back.
        return None

    user_row = get_user_record(user_id)
    if not user_row:
        return None

    supabase_user_id = user_row["supabase_user_id"] if "supabase_user_id" in user_row.keys() else None
    include_practice_time = sync_practice_time or not supabase_user_id

    pg_conn = _get_supabase_connection()
    if not pg_conn:
        return supabase_user_id

    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public.app_users (
                        id uuid NOT NULL DEFAULT uuid_generate_v4(),
                        username text NOT NULL UNIQUE,
                        password_hash text,
                        camera_enabled boolean NOT NULL DEFAULT true,
                        camera_index integer NOT NULL DEFAULT 0,
                        practice_time_seconds integer NOT NULL DEFAULT 0,
                        migrated_at timestamptz NOT NULL DEFAULT now(),
                        CONSTRAINT app_users_pkey PRIMARY KEY (id)
                    );
                    """
                )

                practice_time_seconds = user_row["practice_time_seconds"]
                base_params = (
                    user_row["username"],
                    user_row["password_hash"],
                    bool(user_row["camera_enabled"]),
                    user_row["camera_index"],
                    practice_time_seconds,
                )

                result = None
                if supabase_user_id:
                    if include_practice_time:
                        cur.execute(
                            """
                            UPDATE public.app_users
                            SET username = %s,
                                password_hash = %s,
                                camera_enabled = %s,
                                camera_index = %s,
                                practice_time_seconds = %s,
                                migrated_at = now()
                            WHERE id = %s
                            RETURNING id;
                            """,
                            (*base_params, supabase_user_id),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE public.app_users
                            SET username = %s,
                                password_hash = %s,
                                camera_enabled = %s,
                                camera_index = %s,
                                migrated_at = now()
                            WHERE id = %s
                            RETURNING id;
                            """,
                            base_params[:-1] + (supabase_user_id,),
                        )
                    result = cur.fetchone()

                if not result:
                    if include_practice_time:
                        upsert_sql = """
                        INSERT INTO public.app_users (username, password_hash, camera_enabled, camera_index, practice_time_seconds)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (username) DO UPDATE SET
                            password_hash = EXCLUDED.password_hash,
                            camera_enabled = EXCLUDED.camera_enabled,
                            camera_index = EXCLUDED.camera_index,
                            practice_time_seconds = EXCLUDED.practice_time_seconds,
                            migrated_at = now()
                        RETURNING id;
                        """
                    else:
                        upsert_sql = """
                        INSERT INTO public.app_users (username, password_hash, camera_enabled, camera_index, practice_time_seconds)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (username) DO UPDATE SET
                            password_hash = EXCLUDED.password_hash,
                            camera_enabled = EXCLUDED.camera_enabled,
                            camera_index = EXCLUDED.camera_index,
                            migrated_at = now()
                        RETURNING id;
                        """
                    cur.execute(upsert_sql, base_params)
                    result = cur.fetchone()

                if isinstance(result, dict):
                    supabase_user_id = result.get("id") or supabase_user_id
                elif result:
                    supabase_user_id = result[0] or supabase_user_id
    except Exception:
        return supabase_user_id
    finally:
        try:
            pg_conn.close()
        except Exception:
            pass

    if supabase_user_id and supabase_user_id != user_row["supabase_user_id"]:
        set_supabase_user_id(user_id, str(supabase_user_id))

    return str(supabase_user_id) if supabase_user_id else None


def normalize_username(username: str) -> str:
    return username.strip().lower()


def increment_practice_time(user_id: int, duration_seconds: int) -> None:
    if duration_seconds <= 0:
        return
    conn = get_db_connection()
    conn.execute(
        "UPDATE user SET practice_time_seconds = practice_time_seconds + ? WHERE id = ?",
        (duration_seconds, user_id),
    )
    conn.commit()
    conn.close()


def increment_hand_sign_count(user_id: int, letter: str) -> None:
    if not letter:
        return
    normalized = letter.strip().upper()
    if len(normalized) != 1 or not normalized.isalpha():
        return
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO hand_sign_stats (user_id, letter, practice_count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, letter) DO UPDATE SET practice_count = practice_count + 1
        """,
        (user_id, normalized),
    )
    conn.commit()
    conn.close()


def get_top_users(limit: int = 5) -> list[dict]:
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT username, practice_time_seconds FROM user "
        "ORDER BY practice_time_seconds DESC, username ASC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {"username": row["username"], "practice_time_seconds": row["practice_time_seconds"]}
        for row in rows
    ]


def get_user_hand_sign_stats(user_id: int) -> list[dict]:
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT letter, practice_count FROM hand_sign_stats WHERE user_id = ? "
        "ORDER BY practice_count DESC, letter ASC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [{"letter": row["letter"], "practice_count": row["practice_count"]} for row in rows]


def log_model_version_local(
    model_type: str,
    version: str,
    notes: str,
    model_filename: str,
    labels_filename: str,
    backup_path: str,
    created_by: str,
) -> None:
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO model_versions
            (model_type, version, notes, model_filename, labels_filename, backup_path, created_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (model_type, version, notes, model_filename, labels_filename, backup_path, created_by),
    )
    conn.commit()
    conn.close()


def log_model_version(
    model_type: str,
    version: str,
    notes: str,
    model_filename: str,
    labels_filename: str,
    backup_path: str,
    created_by: str,
    ) -> None:
    log_model_version_local(model_type, version, notes, model_filename, labels_filename, backup_path, created_by)
    supabase_log_model_version(model_type, version, notes, model_filename, labels_filename, backup_path, created_by)


def get_current_model_versions() -> dict:
    """Fetch the latest recorded versions for dynamic and spelling models."""
    versions = {"dynamic": "not set", "spelling": "not set"}
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = get_db_connection()
        for model_type in ("dynamic", "spelling"):
            row = conn.execute(
                "SELECT version FROM model_versions WHERE model_type = ? ORDER BY id DESC LIMIT 1",
                (model_type,),
            ).fetchone()
            if row and row["version"]:
                versions[model_type] = row["version"]
    except sqlite3.Error:
        # If the table is missing or unreadable, fall back to defaults.
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return versions


def get_latest_model_meta() -> dict:
    """
    Return latest version and notes for dynamic and spelling models.
    """
    meta = {
        "dynamic": {"version": "not set", "notes": ""},
        "spelling": {"version": "not set", "notes": ""},
    }
    try:
        conn = get_db_connection()
        for model_type in ("dynamic", "spelling"):
            row = conn.execute(
                "SELECT version, notes FROM model_versions WHERE model_type = ? ORDER BY id DESC LIMIT 1",
                (model_type,),
            ).fetchone()
            if row:
                meta[model_type]["version"] = row["version"] or "not set"
                meta[model_type]["notes"] = row["notes"] or ""
    except sqlite3.Error:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return meta


def list_model_backups() -> dict:
    """
    Return available backups for each model type.

    Each backup entry only needs a name for the dropdown.
    """
    dynamic_backups: list[dict] = []
    spelling_backups: list[dict] = []
    if not os.path.isdir(MODEL_BACKUP_DIR):
        return {"dynamic": dynamic_backups, "spelling": spelling_backups}

    for name in sorted(os.listdir(MODEL_BACKUP_DIR), reverse=True):
        backup_path = os.path.join(MODEL_BACKUP_DIR, name)
        if not os.path.isdir(backup_path):
            continue
        files = {f.lower() for f in os.listdir(backup_path)}
        entry = {"name": name}
        if {"dynamic_model.h5", "dynamic_labels.txt"}.issubset(files):
            dynamic_backups.append(entry)
        spelling_h5_candidates = {"atoz.h5", "static.h5"}
        if files.intersection(spelling_h5_candidates) and "labels.txt" in files:
            spelling_backups.append(entry)

    return {"dynamic": dynamic_backups, "spelling": spelling_backups}


def infer_backup_model_type(backup_name: str) -> Optional[str]:
    """Infer whether a backup folder contains a dynamic or spelling model."""
    if not backup_name:
        return None
    backup_dir = os.path.join(MODEL_BACKUP_DIR, backup_name)
    if not os.path.isdir(backup_dir):
        return None
    files = {f.lower() for f in os.listdir(backup_dir)}
    has_dynamic = {"dynamic_model.h5", "dynamic_labels.txt"}.issubset(files)
    spelling_h5_candidates = {"atoz.h5", "static.h5"}
    has_spelling = bool(files.intersection(spelling_h5_candidates) and "labels.txt" in files)
    if has_dynamic and not has_spelling:
        return "dynamic"
    if has_spelling and not has_dynamic:
        return "spelling"
    return None


def restore_model_from_backup(model_type: str, backup_name: str) -> None:
    """
    Restore the requested model/labels from a backup folder.

    Uses copy to preserve the backup; raises if any required file is missing.
    """
    backup_dir = os.path.join(MODEL_BACKUP_DIR, backup_name)
    if not os.path.isdir(backup_dir):
        raise FileNotFoundError(f"Backup folder not found: {backup_name}")

    model_path, labels_path = resolve_model_paths(model_type)
    required_files = {
        os.path.join(backup_dir, os.path.basename(model_path)),
        os.path.join(backup_dir, os.path.basename(labels_path)),
    }
    missing = [p for p in required_files if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Backup is missing files: {', '.join(os.path.basename(m) for m in missing)}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def _safe_restore(src: str, dest: str) -> None:
        tmp_dest = f"{dest}.restore_tmp"
        if os.path.exists(tmp_dest):
            os.remove(tmp_dest)
        shutil.copy2(src, tmp_dest)
        os.replace(tmp_dest, dest)

    _safe_restore(os.path.join(backup_dir, os.path.basename(model_path)), model_path)
    _safe_restore(os.path.join(backup_dir, os.path.basename(labels_path)), labels_path)


def has_allowed_extension(filename: str, allowed_ext: set[str]) -> bool:
    return bool(filename and "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext)


def resolve_model_paths(model_type: str) -> tuple[str, str]:
    if model_type == "spelling":
        return SPELLING_MODEL_PATH, SPELLING_LABELS_PATH
    return DYNAMIC_MODEL_PATH, DYNAMIC_LABELS_PATH


def backup_existing_model_files(model_type: str) -> str:
    """Backup current model and labels into a timestamped folder; returns backup path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(MODEL_BACKUP_DIR, f"{timestamp}_{model_type}")
    os.makedirs(backup_dir, exist_ok=True)
    model_path, labels_path = resolve_model_paths(model_type)
    for src in (model_path, labels_path):
        if os.path.exists(src):
            shutil.move(src, os.path.join(backup_dir, os.path.basename(src)))
    return backup_dir


def reload_model_from_disk(model_type: str) -> None:
    """Reload the in-memory keras models/labels after an update."""
    global dynamic_model, spelling_model, dynamic_labels, spelling_labels
    global dynamic_sequence_len, dynamic_img_size, dynamic_channels
    global spelling_img_size, spelling_channels, imgSize, sequence_buffer

    if model_type == "dynamic":
        dynamic_model = _load_model_safe(DYNAMIC_MODEL_PATH)
        dynamic_labels = _load_labels(DYNAMIC_LABELS_PATH) or ["UNKNOWN"]
        dynamic_sequence_len, dynamic_img_size, dynamic_channels = _get_model_shape(dynamic_model, imgSize)
        imgSize = dynamic_img_size
        sequence_buffer = deque(maxlen=dynamic_sequence_len or 30)
    else:
        spelling_model = _load_model_safe(SPELLING_MODEL_PATH)
        spelling_labels = _load_labels(SPELLING_LABELS_PATH) or ["UNKNOWN"]
        _, spelling_img_size, spelling_channels = _get_model_shape(spelling_model, imgSize)


def save_uploaded_model_files(model_type: str, model_file, labels_file) -> tuple[str, str]:
    """Save uploaded model/labels to their target locations."""
    model_path, labels_path = resolve_model_paths(model_type)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_file.save(model_path)
    labels_file.save(labels_path)
    return model_path, labels_path


def dataset_upload_enabled_for_current_user() -> bool:
    if session.get("is_admin"):
        return True
    return get_feature_flag(FEATURE_FLAG_DATASET_UPLOAD, False)


def build_nav_links() -> list[dict]:
    links = [
        {"endpoint": "index", "label": "Dashboard"},
        {"endpoint": "profile", "label": "Profile"},
        {"endpoint": "settings", "label": "Settings"},
    ]
    if dataset_upload_enabled_for_current_user():
        links.append({"endpoint": "dataset_upload", "label": "Dataset Upload"})
    if session.get("is_admin"):
        links.append({"endpoint": "admin_dashboard", "label": "Admin"})
    return links


@app.context_processor
def inject_nav_links():
    return {
        "nav_links": build_nav_links(),
        "dataset_upload_enabled": dataset_upload_enabled_for_current_user(),
        "latest_model_meta": get_latest_model_meta(),
    }


def detect_windows_camera_names() -> list[str]:
    if platform.system() != "Windows":
        return []
    script = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.Service -eq 'usbvideo' -or $_.PNPClass -eq 'Camera' } | "
        "Select-Object -ExpandProperty Name"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


camera_cache: dict = {"ts": 0.0, "data": []}


def open_camera(index: int) -> tuple[Optional[cv2.VideoCapture], Optional[int]]:
    """Try to open a camera index using a few backends (helps on Windows)."""
    backends = [cv2.CAP_ANY]
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    chosen_backend: Optional[int] = None
    cap: Optional[cv2.VideoCapture] = None

    for backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend)
        except TypeError:
            # Older OpenCV builds do not accept the backend argument
            cap = cv2.VideoCapture(index)
        if cap is None:
            continue
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            chosen_backend = backend
            break
        cap.release()
        cap = None

    return cap, chosen_backend


def get_camera_options(max_devices: int = 3) -> list[dict]:
    now = time.time()
    if camera_cache.get("data") and now - camera_cache.get("ts", 0) < 10:
        return camera_cache["data"]
    friendly_names = detect_windows_camera_names()
    cameras: list[dict] = []
    for idx in range(max_devices):
        cap, backend = open_camera(idx)
        if cap and cap.isOpened():
            label = friendly_names[idx] if idx < len(friendly_names) else f"Camera {idx}"
            cameras.append({"index": idx, "label": label})
            cap.release()
    camera_cache["data"] = cameras
    camera_cache["ts"] = now
    return cameras


def get_camera_status() -> dict:
    """Return a small snapshot of camera availability for debugging."""
    available = get_camera_options()
    return {
        "active": camera_active,
        "current_index": current_camera_index,
        "available": available,
    }


def find_first_working_camera(max_devices: int = 5, min_mean: float = 1.0) -> tuple[Optional[cv2.VideoCapture], Optional[int]]:
    """Return an opened camera and its index that can actually return frames."""
    for idx in range(max_devices):
        cap, backend = open_camera(idx)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0 and frame.mean() > min_mean:
            return cap, idx
        cap.release()
    return None, None


# -----------------------------
# Dataset helpers
# -----------------------------

# Initialize detector and model
detector = HandDetector(maxHands=2)
dynamic_model = _load_model_safe(DYNAMIC_MODEL_PATH)
spelling_model = _load_model_safe(SPELLING_MODEL_PATH)

def _normalize_label_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    parts = stripped.split()
    if parts and parts[0].isdigit():
        return " ".join(parts[1:])
    return stripped


def _load_labels(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [
            _normalize_label_line(line)
            for line in f
            if _normalize_label_line(line)
        ]

dynamic_labels = _load_labels(DYNAMIC_LABELS_PATH)
if not dynamic_labels:
    dynamic_labels = ["UNKNOWN"]

spelling_labels = _load_labels(SPELLING_LABELS_PATH)
if not spelling_labels:
    spelling_labels = ["UNKNOWN"]

# Global variables
def _get_model_shape(
    model: tf.keras.Model,
    fallback_img_size: int,
    fallback_seq_len: int = 30,
) -> tuple[Optional[int], int, int]:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if not shape:
        return None, fallback_img_size, 1
    if len(shape) == 5:
        seq_len = shape[1] or fallback_seq_len
        img_size = shape[2] or fallback_img_size
        channels = shape[4] or 1
        return int(seq_len), int(img_size), int(channels)
    if len(shape) == 4:
        img_size = shape[1] or fallback_img_size
        channels = shape[3] or 1
        return None, int(img_size), int(channels)
    return None, fallback_img_size, 1


offset = 20
imgSize = 300
dynamic_sequence_len, dynamic_img_size, dynamic_channels = _get_model_shape(dynamic_model, imgSize)
_, spelling_img_size, spelling_channels = _get_model_shape(spelling_model, imgSize)
imgSize = dynamic_img_size

sequence_buffer = deque(maxlen=dynamic_sequence_len or 30)
spelling_smooth_window: deque = deque(maxlen=SPELLING_SMOOTH_WINDOW)
current_prediction = ""
confidence_scores = []
practice_word = ""
practice_targets: list[str] = []
practice_display_sequence: list[str] = []
current_letter_index = 0
practice_active = False
practice_mode = "spelling"
current_prediction_index = 0
last_inference_mode = practice_mode

# Global camera variables
camera_active = True
current_camera_index = 0


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login as admin to continue.', 'error')
            return redirect(url_for('login'))
        profile = get_user_profile(session['user_id'])
        if not profile or not profile.get('is_admin'):
            flash('Admin access required.', 'error')
            return redirect(url_for('index'))
        session['is_admin'] = True
        return f(*args, **kwargs)

    return decorated_function


def _prepare_frame_for_model(img: np.ndarray, channels: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype("float32") / 255.0
    if channels <= 1:
        return normalized[..., None]
    return np.repeat(normalized[..., None], channels, axis=2)


def _prepare_spelling_frame_for_model(img: np.ndarray, channels: int, normalize: bool = True) -> np.ndarray:
    """
    Spelling preprocessing: keep RGB for multi-channel models; normalize to 0-1 if requested.
    """
    if channels <= 1:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
    else:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = frame.astype("float32")
    if normalize:
        frame = frame / 255.0
    return frame


def legacy_prepare_spelling_frame(
    img: np.ndarray,
    hands: list[dict],
    offset_px: int,
    canvas_size: int,
    model_target_size: int,
    normalize: bool,
    channels: int,
) -> Optional[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """
    Legacy preprocessing: single-hand bbox + offset, white padding to square, optional resize to
    the model target size, keep 3-channel RGB (or grayscale if the model requires 1 channel).
    """
    if not hands:
        return None
    x, y, w, h = hands[0].get("bbox", (0, 0, 0, 0))
    if w <= 0 or h <= 0:
        return None

    img_h, img_w = img.shape[:2]
    x1 = max(0, x - offset_px)
    y1 = max(0, y - offset_px)
    x2 = min(img_w, x + w + offset_px)
    y2 = min(img_h, y + h + offset_px)

    imgWhite = np.ones((canvas_size, canvas_size, 3), np.uint8) * 255
    imgCrop = img[y1:y2, x1:x2]
    if imgCrop.size == 0:
        return None

    aspectRatio = h / w if w else 1.0
    if aspectRatio > 1:
        k = canvas_size / h
        wCal = min(canvas_size, math.ceil(k * w))
        if wCal <= 0:
            return None
        imgResize = cv2.resize(imgCrop, (wCal, canvas_size))
        wGap = max(0, math.ceil((canvas_size - wCal) / 2))
        end_x = min(canvas_size, wGap + wCal)
        imgWhite[:, wGap:end_x] = imgResize[:, : end_x - wGap]
    else:
        k = canvas_size / w
        hCal = min(canvas_size, math.ceil(k * h))
        if hCal <= 0:
            return None
        imgResize = cv2.resize(imgCrop, (canvas_size, hCal))
        hGap = max(0, math.ceil((canvas_size - hCal) / 2))
        end_y = min(canvas_size, hGap + hCal)
        imgWhite[hGap:end_y, :] = imgResize[: end_y - hGap, :]

    if model_target_size and model_target_size != canvas_size:
        imgWhite = cv2.resize(imgWhite, (model_target_size, model_target_size))

    frame = _prepare_spelling_frame_for_model(imgWhite, channels, normalize)
    return frame, (x1, y1, x2, y2)


def _label_threshold(idx: int, labels: list[str]) -> float:
    """Return the confidence threshold for a given predicted index."""
    if not labels or idx < 0 or idx >= len(labels):
        return SPELLING_CONFIDENCE_THRESHOLD_DEFAULT
    label = (labels[idx] or "").strip().upper()
    if label in SPELLING_CONFUSED_LABELS:
        return SPELLING_CONFIDENCE_THRESHOLD_CONFUSED
    return SPELLING_CONFIDENCE_THRESHOLD_DEFAULT


def smooth_spelling_prediction(probs: Optional[list[float]], labels: Optional[list[str]] = None) -> tuple[Optional[int], float, list[float]]:
    """
    Smooth spelling predictions over a small window and enforce a confidence floor
    with stricter rules for easily-confused letters.
    """
    global spelling_smooth_window
    if labels is None:
        labels = spelling_labels
    if probs is None:
        spelling_smooth_window.clear()
        return None, 0.0, []
    arr = np.array(probs, dtype=float)
    spelling_smooth_window.append(arr)
    avg = np.mean(spelling_smooth_window, axis=0)
    if avg.size == 0:
        return None, 0.0, []

    idx = int(np.argmax(avg))
    conf = float(avg[idx])
    threshold = _label_threshold(idx, labels)

    # Runner-up gap check for confused letters.
    sorted_idx = np.argsort(avg)
    runner_idx = int(sorted_idx[-2]) if avg.size >= 2 else idx
    runner_conf = float(avg[runner_idx]) if avg.size >= 2 else 0.0
    gap = conf - runner_conf

    label = (labels[idx] or "").strip().upper() if idx < len(labels) else ""
    if (label in SPELLING_CONFUSED_LABELS and gap < SPELLING_CONFUSED_GAP) or conf < threshold:
        return None, conf, avg.tolist()

    return idx, conf, avg.tolist()


def generate_frames():
    global current_prediction, confidence_scores, camera_active, current_camera_index
    global current_prediction_index, last_inference_mode

    cap: Optional[cv2.VideoCapture] = None
    active_camera_index: Optional[int] = None
    active_backend: Optional[int] = None

    try:
        while True:
            try:
                if practice_mode != last_inference_mode:
                    sequence_buffer.clear()
                    last_inference_mode = practice_mode

                if not camera_active:
                    # Return a black frame with "Camera Disabled" message
                    if cap is not None:
                        cap.release()
                        cap = None
                        active_camera_index = None
                        active_backend = None

                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_frame, "Camera Disabled", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', black_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.15)
                    continue

                # Check if camera should be active and reinitialize if needed
                if (cap is None or not cap.isOpened()) or (active_camera_index != current_camera_index):
                    if cap is not None:
                        cap.release()
                        cap = None

                    cap, active_backend = open_camera(current_camera_index)
                    active_camera_index = current_camera_index if cap else None

                    if cap is None or not cap.isOpened():
                        # Try fallbacks automatically
                        if cap:
                            cap.release()
                        cap, fallback_idx = find_first_working_camera()
                        if cap and cap.isOpened():
                            current_camera_index = fallback_idx if fallback_idx is not None else current_camera_index
                            active_camera_index = fallback_idx
                            continue

                        # Return a black frame with error message
                        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            black_frame,
                            f"Camera {current_camera_index} not available",
                            (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (255, 255, 255),
                            2,
                        )
                        ret, buffer = cv2.imencode('.jpg', black_frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        time.sleep(0.25)
                        continue

                success, img = cap.read() if cap else (False, None)
                if not success or img is None or img.size == 0 or img.mean() <= 1.0:
                    # Reset and try again next loop iteration with fallback
                    if cap is not None:
                        cap.release()
                    cap, fallback_idx = find_first_working_camera()
                    if cap and cap.isOpened():
                        current_camera_index = fallback_idx if fallback_idx is not None else current_camera_index
                        active_camera_index = fallback_idx
                        continue

                    cap = None
                    active_camera_index = None

                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    msg = "Failed to read from camera" if img is None else "No video frames"
                    cv2.putText(
                        black_frame,
                        f"{msg} {current_camera_index}",
                        (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    ret, buffer = cv2.imencode('.jpg', black_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.1)
                    continue

                imgOutput = img.copy()
                hands, _ = detector.findHands(img, draw=False)

                if hands:
                    use_dynamic = practice_mode == "words"
                    is_legacy_spelling = not use_dynamic and USE_LEGACY_SPELLING_PREPROCESS
                    prediction = None
                    index = 0

                    # Focus on the largest detected hand to keep the ROI tight.
                    primary_hand = max(
                        hands,
                        key=lambda h: (h.get("bbox", (0, 0, 0, 0))[2] * h.get("bbox", (0, 0, 0, 0))[3]),
                    )
                    hands_for_crop = [primary_hand]

                    if use_dynamic:
                        spelling_smooth_window.clear()

                        target_size = dynamic_img_size
                        target_channels = dynamic_channels
                        x1, y1, x2, y2 = expand_hand_bbox(
                            hands_for_crop, img.shape, offset, HAND_CONTEXT_RATIO_DYNAMIC
                        )
                        w = x2 - x1
                        h = y2 - y1

                        imgWhite = np.ones((target_size, target_size, 3), np.uint8) * 255
                        imgCrop = img[y1:y2, x1:x2]
                        if imgCrop.size == 0:
                            continue

                        aspectRatio = h / w
                        if aspectRatio > 1:
                            k = target_size / h
                            wCal = min(target_size, math.ceil(k * w))

                            if wCal > 0:
                                imgResize = cv2.resize(imgCrop, (wCal, target_size))
                                wGap = max(0, math.ceil((target_size - wCal) / 2))
                                end_x = min(target_size, wGap + wCal)
                                imgWhite[:, wGap:end_x] = imgResize[:, : end_x - wGap]
                        else:
                            k = target_size / w
                            hCal = min(target_size, math.ceil(k * h))

                            if hCal > 0:
                                imgResize = cv2.resize(imgCrop, (target_size, hCal))
                                hGap = max(0, math.ceil((target_size - hCal) / 2))
                                end_y = min(target_size, hGap + hCal)
                                imgWhite[hGap:end_y, :] = imgResize[: end_y - hGap, :]
                        frame_input = _prepare_frame_for_model(imgWhite, target_channels)

                        sequence_buffer.append(frame_input)

                        if len(sequence_buffer) == (dynamic_sequence_len or 30):
                            seq = np.stack(sequence_buffer, axis=0)
                            seq = np.expand_dims(seq, axis=0)
                            try:
                                prediction = dynamic_model.predict(seq, verbose=0)[0]
                                index = int(np.argmax(prediction))
                                current_prediction_index = index
                            except Exception:
                                prediction = None
                                current_prediction_index = 0
                        else:
                            prediction = None
                            current_prediction_index = 0

                        # Update global variables after prediction
                        if prediction is not None:
                            current_prediction = (
                                dynamic_labels[index] if index < len(dynamic_labels) else "UNKNOWN"
                            )
                            confidence_scores = prediction if isinstance(prediction, list) else prediction.tolist()
                        else:
                            current_prediction = f"Collecting frames ({len(sequence_buffer)}/{dynamic_sequence_len or 30})"
                            confidence_scores = []
                    else:
                        sequence_buffer.clear()
                        if is_legacy_spelling:
                            legacy = legacy_prepare_spelling_frame(
                                img,
                                hands_for_crop,
                                offset,
                                LEGACY_SPELLING_IMG_SIZE,
                                spelling_img_size,
                                LEGACY_SPELLING_NORMALIZE,
                                spelling_channels,
                            )
                            if legacy is None:
                                continue
                            frame_input, (x1, y1, x2, y2) = legacy
                        else:
                            target_size = spelling_img_size
                            target_channels = spelling_channels
                            x1, y1, x2, y2 = expand_hand_bbox(
                                hands_for_crop, img.shape, offset, HAND_CONTEXT_RATIO_SPELLING
                            )
                            w = x2 - x1
                            h = y2 - y1
                            imgWhite = np.ones((target_size, target_size, 3), np.uint8) * 255
                            imgCrop = img[y1:y2, x1:x2]
                            if imgCrop.size == 0:
                                spelling_smooth_window.clear()
                                continue
                            aspectRatio = h / w
                            if aspectRatio > 1:
                                k = target_size / h
                                wCal = min(target_size, math.ceil(k * w))

                                if wCal > 0:
                                    imgResize = cv2.resize(imgCrop, (wCal, target_size))
                                    wGap = max(0, math.ceil((target_size - wCal) / 2))
                                    end_x = min(target_size, wGap + wCal)
                                    imgWhite[:, wGap:end_x] = imgResize[:, : end_x - wGap]
                            else:
                                k = target_size / w
                                hCal = min(target_size, math.ceil(k * h))

                                if hCal > 0:
                                    imgResize = cv2.resize(imgCrop, (target_size, hCal))
                                    hGap = max(0, math.ceil((target_size - hCal) / 2))
                                    end_y = min(target_size, hGap + hCal)
                                    imgWhite[hGap:end_y, :] = imgResize[: end_y - hGap, :]
                            frame_input = _prepare_spelling_frame_for_model(imgWhite, target_channels, normalize=True)

                        try:
                            raw_prediction = spelling_model.predict(
                                np.expand_dims(frame_input, axis=0), verbose=0
                            )[0]
                        except Exception:
                            raw_prediction = None
                        index, conf, averaged = smooth_spelling_prediction(
                            raw_prediction if isinstance(raw_prediction, (list, np.ndarray)) else None,
                            spelling_labels,
                        )
                        if index is None:
                            current_prediction_index = 0
                            current_prediction = "Hold still"
                        else:
                            current_prediction_index = index
                            current_prediction = (
                                spelling_labels[index] if index < len(spelling_labels) else "UNKNOWN"
                            )
                        confidence_scores = averaged

                    # Draw bounding boxes and labels
                    cv2.rectangle(imgOutput, (x1, max(0, y1 - 50)),
                                  (x1 + 180, y1), (255, 0, 255), cv2.FILLED)

                    cv2.putText(imgOutput, current_prediction, (x1 + 5, max(0, y1 - 15)),
                                cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

                    cv2.rectangle(imgOutput, (x1, y1),
                                  (x2, y2), (255, 0, 255), 4)
                else:
                    current_prediction = "No hand detected"
                    confidence_scores = []
                    sequence_buffer.clear()
                    spelling_smooth_window.clear()

                # Encode frame
                ret, buffer = cv2.imencode('.jpg', imgOutput)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as exc:
                print(f"[camera] error: {exc}", flush=True)
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "Camera error", (160, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(black_frame, str(exc)[:40], (40, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.25)
    finally:
        if cap is not None:
            cap.release()


@app.route('/')
def home():
    # Always start at the login endpoint; the login view itself will bounce
    # authenticated users to the dashboard.
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('register.html')

        if len(username) < 3:
            flash('Username must be at least 3 characters long', 'error')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')

        # Check if user already exists
        if username_exists(username):
            flash('Username already exists', 'error')
            return render_template('register.html')

        try:
            create_user(username, password)
        except ValueError:
            flash('Username already exists', 'error')
            return render_template('register.html')
        except sqlite3.Error:
            flash('An error occurred while creating the account. Please try again.', 'error')
            return render_template('register.html')

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        global camera_active, current_camera_index, practice_mode, practice_active
        global practice_word, practice_targets, practice_display_sequence, current_letter_index
        global sequence_buffer, spelling_smooth_window
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('login.html')

        profile = verify_user_credentials(username, password)
        if profile:
            session['user_id'] = profile['id']
            session['username'] = profile['username']
            session['is_admin'] = bool(profile.get('is_admin'))
            camera_active = profile.get('camera_enabled', True)
            current_camera_index = profile.get('camera_index', 0)
            practice_mode = "spelling"
            practice_active = False
            practice_word = ""
            practice_targets = []
            practice_display_sequence = []
            current_letter_index = 0
            sequence_buffer.clear()
            spelling_smooth_window.clear()
            flash('Login successful!', 'success')
            return redirect(url_for('index'))

        flash('Invalid username or password', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def index():
    global camera_active, current_camera_index
    global practice_mode, practice_active, practice_word, practice_targets, practice_display_sequence, current_letter_index
    global sequence_buffer, spelling_smooth_window
    uid = session.get('user_id')
    if not uid:
        session.clear()
        flash('Session expired. Please login again.', 'error')
        return redirect(url_for('login'))
    profile = get_user_profile(uid)
    if not profile:
        session.clear()
        flash('Unable to load profile information. Please login again.', 'error')
        return redirect(url_for('login'))

    # Default to spelling mode for letters each time the dashboard loads.
    practice_mode = "spelling"
    practice_active = False
    practice_word = ""
    practice_targets = []
    practice_display_sequence = []
    current_letter_index = 0
    sequence_buffer.clear()
    spelling_smooth_window.clear()

    camera_active = profile.get('camera_enabled', True)
    current_camera_index = profile.get('camera_index', 0)
    session['username'] = profile.get('username', session.get('username', ''))
    session['is_admin'] = profile.get('is_admin', False)
    return render_template('dashboard.html', username=session['username'], is_admin=session.get('is_admin', False))


@app.route('/dataset/upload', methods=['GET', 'POST'])
@login_required
def dataset_upload():
    if not dataset_upload_enabled_for_current_user():
        flash('Dataset upload is currently disabled by admin.', 'error')
        if session.get("is_admin"):
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('index'))
    if request.method == 'POST':
        word = safe_word_folder(request.form.get('word', ''))
        sample_fps_raw = request.form.get('sample_fps', '60')
        target_frames_raw = request.form.get('target_frames', '100')
        save_raw = request.form.get('save_raw') == 'on'
        save_processed = request.form.get('save_processed') == 'on'

        if not word:
            flash('Please provide a valid label (word).', 'error')
            return render_template('dataset_upload.html')

        files = [f for f in request.files.getlist('videos') if f and f.filename]
        if not files:
            flash('Please upload at least one video file.', 'error')
            return render_template('dataset_upload.html')

        invalid = [f.filename for f in files if not allowed_file(f.filename, ALLOWED_VIDEO_EXT)]
        if invalid:
            flash(f'Unsupported file type: {", ".join(invalid)}. Use mp4, mov, avi, mkv, or webm.', 'error')
            return render_template('dataset_upload.html')

        try:
            sample_fps = int(sample_fps_raw)
            target_frames = int(target_frames_raw)
        except ValueError:
            flash('Sample FPS and target frames must be integers.', 'error')
            return render_template('dataset_upload.html')

        sample_fps = max(1, sample_fps)
        target_frames = max(0, target_frames)

        os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)
        os.makedirs(BASE_DATASET_DIR, exist_ok=True)

        results = []
        for file in files:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(BASE_UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
            file.save(upload_path)

            try:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
                sample_dir = make_sample_dir(BASE_DATASET_DIR, word, run_id)
                meta = extract_frames_from_video(
                    upload_path,
                    sample_dir,
                    detector,
                    offset,
                    imgSize,
                    sample_fps=sample_fps,
                    target_frames=target_frames,
                    save_raw=save_raw,
                    save_processed=save_processed,
                    extra_ratio=HAND_CONTEXT_RATIO,
                    target_img_size=224,
                )
                results.append({
                    "filename": filename,
                    "sample_dir": sample_dir,
                    "meta": meta,
                })
            except Exception as exc:
                flash(f'Upload processed but failed during extraction for {filename}: {exc}', 'error')

        if results:
            flash(f'Processed {len(results)} video(s) successfully.', 'success')
        return render_template('dataset_upload.html', results=results, word=word)

    return render_template('dataset_upload.html', results=None)


@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin_dashboard():
    action = (request.form.get('action', '') or '').lower() if request.method == 'POST' else ''

    if action == 'toggle_dataset_visibility':
        visible = request.form.get('dataset_visible') == 'on'
        set_feature_flag(FEATURE_FLAG_DATASET_UPLOAD, visible)
        flash(f'Dataset upload visibility set to {"on" if visible else "off"}.', 'success')
        return redirect(url_for('admin_dashboard'))

    if action == 'rollback':
        model_type = (request.form.get('rollback_model_type', '') or '').lower()
        backup_name = (request.form.get('backup_name', '') or '').strip()
        uploader = session.get('username', 'admin')

        if not backup_name:
            flash('Choose a backup to restore.', 'error')
            return redirect(url_for('admin_dashboard'))
        if model_type not in ('dynamic', 'spelling'):
            inferred = infer_backup_model_type(backup_name)
            if inferred:
                model_type = inferred
            else:
                flash('Select a valid model type for rollback.', 'error')
                return redirect(url_for('admin_dashboard'))

        created_backup = ""
        try:
            created_backup = backup_existing_model_files(model_type)
            restore_model_from_backup(model_type, backup_name)
            reload_model_from_disk(model_type)
            log_model_version(
                model_type=model_type,
                version=f"rollback-{backup_name}",
                notes=f"Rollback applied from backup {backup_name}",
                model_filename=os.path.basename(resolve_model_paths(model_type)[0]),
                labels_filename=os.path.basename(resolve_model_paths(model_type)[1]),
                backup_path=os.path.join(MODEL_BACKUP_DIR, backup_name),
                created_by=uploader,
            )
            flash('Rollback completed successfully.', 'success')
        except Exception as exc:
            if created_backup:
                try:
                    restore_model_from_backup(model_type, os.path.basename(created_backup))
                except Exception:
                    pass
            flash(f'Rollback failed: {exc}', 'error')
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        model_type = (request.form.get('model_type', '') or '').lower()
        version = (request.form.get('version', '') or '').strip()
        notes = (request.form.get('notes', '') or '').strip()
        model_file = request.files.get('model_file')
        labels_file = request.files.get('labels_file')
        uploader = session.get('username', 'admin')

        if model_type not in ('dynamic', 'spelling'):
            flash('Select a valid model type (dynamic or spelling).', 'error')
            return redirect(url_for('admin_dashboard'))
        if not version:
            flash('Version is required.', 'error')
            return redirect(url_for('admin_dashboard'))
        if not notes:
            flash('Please provide notes describing the update.', 'error')
            return redirect(url_for('admin_dashboard'))
        if not model_file or not model_file.filename:
            flash('Upload a .h5 model file.', 'error')
            return redirect(url_for('admin_dashboard'))
        if not labels_file or not labels_file.filename:
            flash('Upload a .txt labels file.', 'error')
            return redirect(url_for('admin_dashboard'))
        if not has_allowed_extension(model_file.filename, ALLOWED_MODEL_EXT):
            flash('Model file must be .h5', 'error')
            return redirect(url_for('admin_dashboard'))
        if not has_allowed_extension(labels_file.filename, ALLOWED_LABEL_EXT):
            flash('Labels file must be .txt', 'error')
            return redirect(url_for('admin_dashboard'))

        backup_path = ""
        model_dest = ""
        label_dest = ""
        try:
            backup_path = backup_existing_model_files(model_type)
            model_dest, label_dest = save_uploaded_model_files(model_type, model_file, labels_file)
            reload_model_from_disk(model_type)
            log_model_version(
                model_type=model_type,
                version=version,
                notes=notes,
                model_filename=os.path.basename(model_dest),
                labels_filename=os.path.basename(label_dest),
                backup_path=backup_path,
                created_by=uploader,
            )
            flash('Model and labels updated successfully.', 'success')
            return redirect(url_for('admin_dashboard'))
        except Exception as exc:
            # Attempt to restore from backup if something failed mid-update
            if backup_path:
                model_path, labels_path = resolve_model_paths(model_type)
                for src_path, dest_path in (
                    (os.path.join(backup_path, os.path.basename(model_dest or model_path)), model_path),
                    (os.path.join(backup_path, os.path.basename(label_dest or labels_path)), labels_path),
                ):
                    try:
                        if os.path.exists(src_path):
                            shutil.move(src_path, dest_path)
                    except Exception:
                        pass
            flash(f'Failed to update model: {exc}', 'error')
            return redirect(url_for('admin_dashboard'))

    current_versions = get_current_model_versions()
    backups = list_model_backups()
    dataset_visible = get_feature_flag(FEATURE_FLAG_DATASET_UPLOAD, False)
    return render_template(
        'admin_dashboard.html',
        username=session.get('username', ''),
        is_admin=True,
        current_versions=current_versions,
        dynamic_backups=backups["dynamic"],
        spelling_backups=backups["spelling"],
        dataset_visible=dataset_visible,
    )


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    uid = session.get('user_id')
    if not uid:
        session.clear()
        flash('Session expired. Please login again.', 'error')
        return redirect(url_for('login'))
    user_profile = get_user_profile(uid)
    if not user_profile:
        session.clear()
        flash('Unable to load profile information. Please login again.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_username = request.form.get('username', '').strip()

        if not new_username:
            flash('Username cannot be empty', 'error')
            return render_template('profile.html', user=user_profile)

        if len(new_username) < 3:
            flash('Username must be at least 3 characters long', 'error')
            return render_template('profile.html', user=user_profile)

        if username_exists(new_username, exclude_uid=uid):
            flash('Username already taken', 'error')
            return render_template('profile.html', user=user_profile)

        try:
            update_user_profile(uid, {"username": new_username})
            session['username'] = new_username
            user_profile['username'] = new_username
            flash('Profile updated successfully!', 'success')
        except sqlite3.Error:
            flash('An error occurred. Please try again.', 'error')

    return render_template('profile.html', user=user_profile)


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    global camera_active, current_camera_index
    uid = session.get('user_id')
    if not uid:
        session.clear()
        flash('Session expired. Please login again.', 'error')
        return redirect(url_for('login'))

    user_profile = get_user_profile(uid)
    if not user_profile:
        session.clear()
        flash('Unable to load profile information. Please login again.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'toggle_camera':
            new_state = not user_profile.get('camera_enabled', True)
            try:
                update_user_profile(uid, {"camera_enabled": new_state})
            except sqlite3.Error:
                flash('Failed to update camera state.', 'error')
                return render_template('settings.html',
                                       user=user_profile,
                                       available_cameras=[],
                                       current_camera_label=f"Camera {user_profile.get('camera_index', 0)}")
            user_profile['camera_enabled'] = new_state
            camera_active = new_state
            status = "enabled" if camera_active else "disabled"
            flash(f'Camera {status} successfully!', 'success')
            return redirect(url_for('settings'))

        elif action == 'change_camera':
            try:
                new_index = int(request.form.get('camera_index', 0))
            except (TypeError, ValueError):
                flash('Invalid camera selection.', 'error')
                return redirect(url_for('settings'))
            try:
                update_user_profile(uid, {"camera_index": new_index})
            except sqlite3.Error:
                flash('Failed to update camera selection.', 'error')
                return render_template('settings.html',
                                       user=user_profile,
                                       available_cameras=[],
                                       current_camera_label=f"Camera {user_profile.get('camera_index', 0)}")
            user_profile['camera_index'] = new_index
            current_camera_index = new_index
            flash(f'Camera switched to index {new_index}', 'success')
            return redirect(url_for('settings'))

    # Get available cameras
    available_cameras = get_camera_options()
    current_label = next(
        (cam["label"] for cam in available_cameras if cam["index"] == user_profile.get('camera_index', 0)),
        f"Camera {user_profile.get('camera_index', 0)}"
    )

    return render_template('settings.html',
                           user=user_profile,
                           available_cameras=available_cameras,
                           current_camera_label=current_label)


@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/available_words')
@login_required
def available_words():
    return jsonify({
        'version': "1.0",
        'words': get_hand_sign_words()
    })


@app.route('/camera_status')
@login_required
def camera_status():
    return jsonify(get_camera_status())


@app.route('/get_prediction')
@login_required
def get_prediction():
    global current_letter_index, practice_active, practice_mode

    safe_confidence = [float(score) for score in confidence_scores] if confidence_scores else []
    active_labels = dynamic_labels if practice_mode == "words" else spelling_labels

    should_advance = False
    normalized_prediction = normalize_word_token(current_prediction)
    current_target = ""
    if practice_active and practice_targets and current_letter_index < len(practice_targets):
        current_target = practice_targets[current_letter_index]
        if normalized_prediction and normalized_prediction == current_target:
            should_advance = True

    return jsonify({
        'prediction': current_prediction,
        'normalized_prediction': normalized_prediction,
        'confidence': safe_confidence,
        'labels': active_labels,
        'practice_active': practice_active,
        'practice_mode': practice_mode,
        'practice_word': practice_word,
        'practice_targets': practice_targets,
        'practice_sequence': practice_display_sequence,
        'current_letter_index': current_letter_index,
        'current_target': current_target,
        'should_advance': should_advance
    })


@app.route('/set_practice_word', methods=['POST'])
@login_required
def set_practice_word():
    global practice_word, current_letter_index, practice_active, practice_mode, practice_targets, practice_display_sequence
    data = request.get_json() or {}
    requested_mode = data.get('mode', 'spelling')
    raw_word = str(data.get('word', '')).strip()
    normalized_word = normalize_word_token(raw_word)

    if requested_mode not in ('spelling', 'words'):
        return jsonify({'success': False, 'message': 'Invalid practice mode selected.'}), 400

    if requested_mode == 'spelling':
        if not normalized_word:
            return jsonify({'success': False, 'message': 'Enter a word that only contains letters.'}), 400
        practice_word = normalized_word
        practice_targets = list(practice_word)
        practice_display_sequence = list(practice_word)
    else:
        token_pairs = tokenize_phrase_chunks(raw_word)
        if not token_pairs:
            return jsonify({'success': False, 'message': 'Enter at least one supported word.'}), 400

        entries: list[dict] = []
        missing_tokens: list[str] = []
        for raw_token, canonical in token_pairs:
            entry = find_hand_sign_word_entry(canonical)
            if not entry:
                missing_tokens.append(raw_token.upper())
            else:
                entries.append(entry)

        if missing_tokens:
            missing_list = ", ".join(missing_tokens)
            return jsonify({
                'success': False,
                'message': f"The following words are not available in Version 1.0 yet: {missing_list}"
            }), 400

        practice_word = " ".join(entry['word'].upper() for entry in entries)
        practice_targets = [entry['canonical'] for entry in entries]
        practice_display_sequence = [entry['word'].upper() for entry in entries]

    practice_mode = requested_mode
    current_letter_index = 0
    practice_active = False
    return jsonify({
        'success': True,
        'word': practice_word,
        'mode': practice_mode,
        'targets': practice_targets,
        'sequence': practice_display_sequence,
    })


@app.route('/set_practice_mode', methods=['POST'])
@login_required
def set_practice_mode():
    global practice_word, current_letter_index, practice_active, practice_mode, practice_targets, practice_display_sequence
    data = request.get_json() or {}
    requested_mode = data.get('mode', 'spelling')

    if requested_mode not in ('spelling', 'words'):
        return jsonify({'success': False, 'message': 'Invalid practice mode selected.'}), 400

    practice_mode = requested_mode
    practice_word = ""
    practice_targets = []
    practice_display_sequence = []
    current_letter_index = 0
    practice_active = False
    sequence_buffer.clear()
    return jsonify({'success': True, 'mode': practice_mode})


@app.route('/start_practice', methods=['POST'])
@login_required
def start_practice():
    global practice_active, current_letter_index
    if not practice_targets:
        return jsonify({'success': False, 'message': 'Add a word before starting practice.'}), 400
    practice_active = True
    current_letter_index = 0
    return jsonify({'success': True})


@app.route('/reset_practice', methods=['POST'])
@login_required
def reset_practice():
    global practice_word, current_letter_index, practice_active, practice_targets, practice_display_sequence
    practice_word = ""
    practice_targets = []
    practice_display_sequence = []
    current_letter_index = 0
    practice_active = False
    return jsonify({'success': True})


@app.route('/advance_letter', methods=['POST'])
@login_required
def advance_letter():
    global current_letter_index
    if practice_targets and current_letter_index < len(practice_targets) - 1:
        current_letter_index += 1
    return jsonify({'success': True, 'current_index': current_letter_index})


@app.route('/log_practice_time', methods=['POST'])
@login_required
def log_practice_time():
    uid = session.get('user_id')
    data = request.get_json() or {}
    try:
        duration = int(float(data.get('duration_seconds', 0)))
    except (TypeError, ValueError):
        duration = 0
    if duration <= 0:
        return jsonify({'success': False, 'message': 'Invalid duration'}), 400
    increment_practice_time(uid, duration)
    return jsonify({'success': True})


@app.route('/log_practice_letter', methods=['POST'])
@login_required
def log_practice_letter():
    uid = session.get('user_id')
    data = request.get_json() or {}
    letter = str(data.get('letter', '')).strip().upper()
    if len(letter) != 1 or not letter.isalpha():
        return jsonify({'success': False, 'message': 'Invalid letter'}), 400
    increment_hand_sign_count(uid, letter)
    return jsonify({'success': True})


@app.route('/dashboard_stats')
@login_required
def dashboard_stats():
    uid = session.get('user_id')
    stats = {
        "top_users": get_top_users(),
        "hand_signs": get_user_hand_sign_stats(uid)
    }
    return jsonify(stats)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)

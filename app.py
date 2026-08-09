"""
Databricks App boilerplate for Weather Data:
- Serves a small Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls weather data from Weather API via weather_client.py and syncs it into Lakebase
- Performs semantic vector similarity search via pgvector on weather documents inside the `weather` schema
"""

import logging
import os
import re

import requests
from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

import lakebase
from weather_client import WeatherClient, sync_weather_documents_to_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)
_w = WorkspaceClient()

TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather.weather_records")
LOCATIONS_TABLE_NAME = os.environ.get("LOCATIONS_TABLE_NAME", "weather.saved_locations")
FORECAST_TABLE_NAME = os.environ.get("FORECAST_TABLE_NAME", "weather.forecast_documents")
DOCUMENTS_TABLE_NAME = os.environ.get("WEATHER_DOCUMENTS_TABLE_NAME", "weather.weather_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE_NAME", "weather.weather_embeddings")

# Default cities to fetch weather forecasts for (comma-separated)
DEFAULT_CITIES = [
    c.strip().title()
    for c in os.environ.get("DEFAULT_CITIES", "New York, London, Tokyo, Paris, Sydney").split(",")
    if c.strip()
]

# Validation regex for city name inputs (letters, spaces, commas, hyphens)
_LOCATION_RE = re.compile(r"^[a-zA-Z\s,-]{1,60}$")

# Load the embedding model ONCE at module/app startup (not per request)
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
logger.info("Initializing embedding model: %s", EMBEDDING_MODEL_NAME)
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.warning("Failed to load SentenceTransformer model '%s': %s", EMBEDDING_MODEL_NAME, e)
    embedding_model = None


def ensure_schema_and_tables():
    """Ensure extensions and tables exist in Lakebase."""
    # Attempt schema creation safely
    try:
        lakebase.run_write("CREATE SCHEMA IF NOT EXISTS weather;")
    except Exception as e:
        logger.debug("Schema creation skipped or unpermitted: %s", e)

    try:
        lakebase.run_write("CREATE EXTENSION IF NOT EXISTS vector;")
    except Exception as e:
        logger.debug("Extension creation skipped or unpermitted: %s", e)

    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
            headline TEXT,
            narrative_text TEXT NOT NULL,
            issued_at TIMESTAMPTZ,
            effective_at TIMESTAMPTZ,
            payload JSONB,
            synced_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE_NAME} (
            id BIGSERIAL PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES {DOCUMENTS_TABLE_NAME}(id) ON DELETE CASCADE,
            chunk_index INT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding vector(384) NOT NULL,
            model_name TEXT NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_doc_chunk UNIQUE (document_id, chunk_index)
        );
        """
    )


def ensure_locations_table():
    """Create the user saved locations table in Lakebase if it doesn't exist yet."""
    try:
        lakebase.run_write("CREATE SCHEMA IF NOT EXISTS weather;")
    except Exception as e:
        logger.debug("Schema creation skipped or unpermitted: %s", e)

    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {LOCATIONS_TABLE_NAME} (
            location TEXT NOT NULL,
            email TEXT NOT NULL,
            latest_temp NUMERIC,
            condition TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (location, email)
        );
        """
    )


def init_db():
    """Run database initialization once on application boot."""
    try:
        ensure_schema_and_tables()
        ensure_locations_table()
        logger.info("Database schemas and tables verified.")
    except Exception as e:
        logger.warning(
            "Database initial setup encountered an error (tables/schema may already exist or user lacks DDL rights): %s",
            e,
        )


# Initialize database once at app load time, NOT inside route handlers
init_db()


def _current_user_email() -> str:
    """
    Resolve current user's email for personalized weather locations.
    Uses Databricks header inject or SDK fallback for local testing.
    """
    header_email = request.headers.get("X-Forwarded-Email")
    if header_email:
        return header_email
    return _w.current_user.me().user_name


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON format."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Simple UI to display weather inputs and location syncs."""
    return render_template("index.html")


@app.route("/documents")
def list_documents():
    """Read weather documents already synced into Lakebase."""
    limit = int(request.args.get("limit", 100))
    rows = lakebase.run_query(
        f"""
        SELECT id, location, source_type, headline, narrative_text, issued_at, synced_at 
        FROM {DOCUMENTS_TABLE_NAME} 
        ORDER BY synced_at DESC LIMIT %s
        """,
        (limit,),
    )
    return jsonify(rows)


@app.route("/sync", methods=["POST"])
def sync_harvested_weather():
    """
    Harvest active weather alerts and forecasts for locations 
    and sync them into weather.weather_documents.
    """
    client = WeatherClient()

    body = request.json if request.is_json else {}
    locations = body.get("locations") or DEFAULT_CITIES

    documents = client.harvest(locations)

    with lakebase.get_connection() as conn:
        synced_count = sync_weather_documents_to_db(conn, documents)

    return jsonify({"synced": synced_count, "locations": locations})


@app.route("/locations", methods=["GET"])
def get_saved_locations():
    """Return current user's saved weather locations."""
    email = _current_user_email()
    rows = lakebase.run_query(
        f"SELECT location, email, latest_temp, condition, updated_at FROM {LOCATIONS_TABLE_NAME} "
        f"WHERE email = %s ORDER BY location ASC",
        (email,),
    )
    return jsonify(rows)


@app.route("/locations", methods=["POST"])
def add_saved_location():
    """Fetch current weather for a location and save/update it in Lakebase."""
    if request.is_json:
        location = request.json.get("location", "")
    else:
        location = request.form.get("location", "")

    location = location.strip().title() if isinstance(location, str) else ""

    if not location or not _LOCATION_RE.match(location):
        return jsonify({"error": f"Invalid location format: {location!r}"}), 400

    client = WeatherClient()
    try:
        data = client.get_current_weather(location)
    except Exception:
        # Fallback if standard API endpoint is unavailable
        data = {}

    temp = _extract_temperature(data)
    condition = _extract_condition(data)
    email = _current_user_email()

    lakebase.run_write(
        f"""
        INSERT INTO {LOCATIONS_TABLE_NAME} (location, email, latest_temp, condition, updated_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (location, email) DO UPDATE
            SET latest_temp = EXCLUDED.latest_temp,
                condition = EXCLUDED.condition,
                updated_at = EXCLUDED.updated_at
        """,
        (location, email, temp, condition),
    )

    return jsonify({"location": location, "email": email, "latest_temp": temp, "condition": condition})


@app.route("/locations/<path:location>", methods=["DELETE"])
def delete_saved_location(location: str):
    """Remove a location from the current user's saved list."""
    location = location.strip().title() if isinstance(location, str) else ""
    if not location or not _LOCATION_RE.match(location):
        return jsonify({"error": f"Invalid location format: {location!r}"}), 400

    email = _current_user_email()
    deleted = lakebase.run_write(
        f"DELETE FROM {LOCATIONS_TABLE_NAME} WHERE location = %s AND email = %s",
        (location, email),
    )

    if not deleted:
        return jsonify({"error": f"{location} is not in your saved locations"}), 404

    return jsonify({"location": location, "email": email, "deleted": True})


@app.route("/weather/search", methods=["POST"])
def search_weather():
    """
    Perform cosine similarity semantic search across weather document vector embeddings using pgvector.

    Request Body:
        {"query": "risk of flooding near rivers", "top_k": 5}
    """
    if not request.is_json:
        return jsonify({"error": "Request payload must be JSON"}), 400

    body = request.get_json(silent=True) or {}
    query_str = body.get("query")

    if not query_str or not isinstance(query_str, str) or not query_str.strip():
        return jsonify({"error": "Missing or empty required field 'query'"}), 400

    raw_top_k = body.get("top_k", 5)
    try:
        top_k = int(raw_top_k)
    except (ValueError, TypeError):
        top_k = 5
    top_k = max(1, min(20, top_k))

    if embedding_model is None:
        return jsonify({"error": "Embedding model is unavailable on startup"}), 500

    try:
        vector = embedding_model.encode(query_str.strip()).tolist()
        vector_str = str(vector)
    except Exception as e:
        logger.exception("Failed to compute embedding vector")
        return jsonify({"error": f"Failed to process query vector: {str(e)}"}), 500

    # Query Lakebase using HNSW index and pgvector cosine distance operator <=>
    sql = f"""
        SELECT d.id, d.location, d.headline, d.narrative_text, e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE_NAME} e
        JOIN {DOCUMENTS_TABLE_NAME} d ON d.id = e.document_id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s;
    """

    try:
        rows = lakebase.run_query(sql, (vector_str, vector_str, top_k))
    except Exception as e:
        logger.warning("Vector search query execution failed or table is empty: %s", e)
        return jsonify([]), 200

    if not rows:
        return jsonify([]), 200

    results = []
    for row in rows:
        if isinstance(row, dict):
            results.append({
                "id": row.get("id"),
                "location": row.get("location"),
                "headline": row.get("headline"),
                "chunk_text": row.get("chunk_text"),
                "similarity": float(row.get("similarity")) if row.get("similarity") is not None else 0.0,
            })
        else:
            results.append({
                "id": row[0],
                "location": row[1],
                "headline": row[2],
                "chunk_text": row[4],
                "similarity": float(row[5]) if row[5] is not None else 0.0,
            })

    return jsonify(results)


def _extract_temperature(data: dict) -> float | None:
    """Extract ambient temperature from Weather API payload structure."""
    if not isinstance(data, dict):
        return None

    main = data.get("main", data.get("current", data))
    if isinstance(main, dict):
        for key in ("temp", "temperature", "temp_c", "temp_f"):
            if key in main and main[key] is not None:
                return float(main[key])
    return None


def _extract_condition(data: dict) -> str | None:
    """Extract weather condition text (e.g., 'Sunny', 'Rain') from payload."""
    if not isinstance(data, dict):
        return None

    weather = data.get("weather") or data.get("current", {}).get("condition")
    if isinstance(weather, list) and len(weather) > 0:
        return weather[0].get("main") or weather[0].get("description")
    elif isinstance(weather, dict):
        return weather.get("text") or weather.get("main")
    return None


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    print(f"Flask app running on http://{host}:{port}")
    app.run(debug=True, host=host, port=port)
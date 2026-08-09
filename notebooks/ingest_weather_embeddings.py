# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather -> Vector Embeddings (Lakebase)
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install all required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers trafilatura requests pandas
# MAGIC %pip install -U sentence-transformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC

# COMMAND ----------

dbutils.widgets.text("schema_name", "weather", "Database schema name")
dbutils.widgets.text("documents_table_name", "weather.weather_documents", "Destination table (raw weather docs)")
dbutils.widgets.text("embeddings_table_name", "weather.weather_embeddings", "Destination table (vectors)")
dbutils.widgets.text("locations", "Chicago, IL; Austin, TX", "Semicolon-separated locations (City, ST or lat,lon)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("nws_api_base_url", "https://api.weather.gov", "NWS API base URL")
dbutils.widgets.text("user_agent", "WeatherLakebaseApp/1.0 (contact@example.com)", "NWS API User-Agent header")
dbutils.widgets.text("weather_fetch_limit", "50", "Max alerts/forecast periods to fetch per location")
dbutils.widgets.text("chunk_size", "800", "Narrative content chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Narrative content chunk overlap (chars)")

SCHEMA_NAME = dbutils.widgets.get("schema_name")
DOCUMENTS_TABLE_NAME = dbutils.widgets.get("documents_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
LOCATIONS = [loc.strip() for loc in dbutils.widgets.get("locations").split(";") if loc.strip()]
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
NWS_API_BASE_URL = dbutils.widgets.get("nws_api_base_url")
USER_AGENT = dbutils.widgets.get("user_agent")
WEATHER_FETCH_LIMIT = int(dbutils.widgets.get("weather_fetch_limit"))
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Rather than hardcoding
# one dimension, switch on the model name so swapping EMBEDDING_MODEL_NAME via
# the widget above automatically resizes the destination table's vector column.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case "text-embedding-3-small":
        EMBEDDING_DIM = 1536
    case "text-embedding-3-large":
        EMBEDDING_DIM = 3072
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")
print(f"Targeting schema: {SCHEMA_NAME!r} | Locations: {LOCATIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single base64-encoded
# MAGIC Postgres URL (`postgresql://role:password@host:5432/db?sslmode=require`)
# MAGIC stored in a Databricks secret scope. We parse it into the pieces psycopg3
# MAGIC needs for connection (host/port/dbname/user/password).

# COMMAND ----------

# DBTITLE 1,Parse Lakebase Connection Info
import base64
from urllib.parse import urlparse, parse_qs
from databricks.sdk import WorkspaceClient

# Configurable widget parameters for secrets management
dbutils.widgets.text("lakebase_secret_scope", "database", "Lakebase secret scope")
dbutils.widgets.text("lakebase_secret_key", "lakebase-url", "Lakebase secret key")

SECRET_SCOPE = dbutils.widgets.get("lakebase_secret_scope")
SECRET_KEY = dbutils.widgets.get("lakebase_secret_key")

w = WorkspaceClient()

def get_lakebase_url(scope: str = SECRET_SCOPE, key: str = SECRET_KEY) -> str:
    """
    Retrieves the PostgreSQL connection string from Databricks Secrets.
    Handles raw connection strings (postgresql://...) as well as base64-encoded secrets.
    """
    try:
        secret_val = dbutils.secrets.get(scope=scope, key=key)
    except Exception:
        secret_obj = w.secrets.get_secret(scope=scope, key=key)
        secret_val = secret_obj.value

    # Clean whitespace or trailing newlines
    secret_val = secret_val.strip()

    # Case 1: Secret is stored directly as a plaintext postgres URL
    if secret_val.startswith("postgresql://") or secret_val.startswith("postgres://"):
        return secret_val

    # Case 2: Secret is base64 encoded -> apply padding fix if needed and decode
    missing_padding = len(secret_val) % 4
    if missing_padding:
        secret_val += "=" * (4 - missing_padding)

    try:
        decoded = base64.b64decode(secret_val).decode("utf-8").strip()
        if decoded.startswith("postgresql://") or decoded.startswith("postgres://"):
            return decoded
    except Exception:
        pass

    return secret_val

# 1. Resolve connection string safely
lakebase_url = get_lakebase_url()
parsed_url = urlparse(lakebase_url)
query_params = parse_qs(parsed_url.query)

# 2. Extract connection components
db_host = parsed_url.hostname
db_port = parsed_url.port or 5432
db_name = parsed_url.path.lstrip("/")
db_user = parsed_url.username
db_password = parsed_url.password
sslmode = query_params.get("sslmode", ["require"])[0]

# 3. Construct psycopg connection parameter dictionary
connection_kwargs = {
    "host": db_host,
    "port": db_port,
    "dbname": db_name,
    "user": db_user,
    "password": db_password,
    "sslmode": sslmode,
}

print("Connection details resolved successfully:")
print(f"  Host:     {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User:     {db_user}")
print(f"  SSL Mode: {sslmode}")

# COMMAND ----------

# DBTITLE 1,Test Psycopg2 connection
import psycopg2
import traceback

print(f"Testing connection to {db_host}:{db_port}/{db_name}")
print(f"Using secret credentials as user: {db_user}\n")

# Test psycopg2 connection
try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode=sslmode,
        connect_timeout=10
    )
    cursor = conn.cursor()
    
    # Query row count from weather documents table
    cursor.execute(f"SELECT COUNT(*) FROM {DOCUMENTS_TABLE_NAME}")
    count = cursor.fetchone()[0]
    print(f"✅ Connection successful! Found {count} rows in {DOCUMENTS_TABLE_NAME}")
    
    # Preview up to 5 documents
    cursor.execute(f"SELECT * FROM {DOCUMENTS_TABLE_NAME} LIMIT 5")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    print(f"\nColumns: {colnames}")
    for row in rows:
        print(row)
    
    cursor.close()
    conn.close()
    print("\n✅ psycopg2 connection and schema query working correctly!")

except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch weather alerts and forecasts from NWS for configured locations
# MAGIC

# COMMAND ----------

# DBTITLE 1,Fetch news and sync using Lakebase SDK
import hashlib
import json
import re
import time
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values
import requests

# Pre-mapped fallback lookup for common locations
CITY_COORDINATES = {
    "CHICAGO, IL": (41.8781, -87.6298),
    "AUSTIN, TX": (30.2672, -97.7431),
    "NEW YORK, NY": (40.7128, -74.0060),
    "MIAMI, FL": (25.7617, -80.1918),
    "SEATTLE, WA": (47.6062, -122.3321),
    "DENVER, CO": (39.7392, -104.9903),
    "LOS ANGELES, CA": (34.0522, -118.2437),
}


def resolve_location(location_input: str) -> tuple[float, float]:
    """Resolves 'lat,lon' string or 'City, ST' to latitude and longitude."""
    cleaned = location_input.strip()

    # Direct lat,lon parsing
    coord_match = re.match(
        r"^([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)$", cleaned
    )
    if coord_match:
        return float(coord_match.group(1)), float(coord_match.group(2))

    # Pre-mapped lookup
    lookup_key = cleaned.upper()
    if lookup_key in CITY_COORDINATES:
        return CITY_COORDINATES[lookup_key]

    # Geocoding fallback via Nominatim OSM API
    try:
        geo_url = "https://nominatim.openstreetmap.org/search"
        params = {"q": cleaned, "format": "json", "limit": 1}
        resp = requests.get(
            geo_url,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as exc:
        print(f"Geocoding fallback failed for '{cleaned}': {exc}")

    raise ValueError(f"Unable to resolve coordinates for location: '{location_input}'")


def fetch_weather_for_location(session: requests.Session, location: str, limit: int) -> list[dict]:
    """Fetches active weather alerts and detailed multi-day forecasts for a single location."""
    docs = []
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    
    lat, lon = resolve_location(location)
    
    # 1. Resolve Gridpoint & Forecast URL
    points_url = f"{NWS_API_BASE_URL}/points/{round(lat, 4)},{round(lon, 4)}"
    p_resp = session.get(points_url, headers=headers, timeout=10)
    p_resp.raise_for_status()
    properties = p_resp.json().get("properties", {})
    forecast_url = properties.get("forecast")

    # 2. Fetch Active Weather Alerts for grid point area
    alerts_url = f"{NWS_API_BASE_URL}/alerts/active?point={round(lat, 4)},{round(lon, 4)}"
    try:
        alert_resp = session.get(alerts_url, headers=headers, timeout=10)
        if alert_resp.status_code == 200:
            features = alert_resp.json().get("features", [])
            for feat in features[:limit]:
                props = feat.get("properties", {})
                alert_id = props.get("id") or hashlib.sha256(f"{location}_{props.get('sent')}".encode()).hexdigest()
                
                desc = (props.get("description") or "").strip()
                inst = (props.get("instruction") or "").strip()
                narrative = f"{desc}\n\nInstructions:\n{inst}".strip() if inst else desc
                
                if narrative:
                    docs.append({
                        "id": f"alert_{alert_id}",
                        "location": location,
                        "source_type": "alert",
                        "headline": props.get("headline") or props.get("event") or "Weather Alert",
                        "narrative_text": narrative,
                        "issued_at": props.get("sent"),
                        "effective_at": props.get("effective"),
                        "payload": json.dumps(feat),
                        "synced_at": datetime.now(timezone.utc).isoformat()
                    })
    except Exception as exc:
        print(f"Warning: Failed to fetch alerts for {location}: {exc}")

    # 3. Fetch Forecast Discussion Narratives
    try:
        if forecast_url:
            fc_resp = session.get(forecast_url, headers=headers, timeout=10)
            if fc_resp.status_code == 200:
                periods = fc_resp.json().get("properties", {}).get("periods", [])
                for period in periods[:limit]:
                    narrative = (period.get("detailedForecast") or "").strip()
                    period_name = period.get("name", "Period")
                    issued_str = period.get("startTime")
                    
                    # Deduplication key for forecast period
                    dedup_key = f"{location}_{period_name}_{issued_str}"
                    doc_id = hashlib.sha256(dedup_key.encode()).hexdigest()[:24]
                    
                    if narrative:
                        docs.append({
                            "id": f"forecast_{doc_id}",
                            "location": location,
                            "source_type": "forecast",
                            "headline": f"{location} — {period_name}: {period.get('shortForecast')}",
                            "narrative_text": narrative,
                            "issued_at": issued_str,
                            "effective_at": period.get("endTime"),
                            "payload": json.dumps(period),
                            "synced_at": datetime.now(timezone.utc).isoformat()
                        })
    except Exception as exc:
        print(f"Warning: Failed to fetch forecast for {location}: {exc}")

    return docs


def upsert_weather_documents_to_lakebase(documents: list[dict]) -> int:
    """Upserts normalized weather documents into Lakebase via psycopg2."""
    if not documents:
        return 0

    upsert_sql = f"""
        INSERT INTO {DOCUMENTS_TABLE_NAME} (
            id, location, source_type, headline, narrative_text, issued_at, effective_at, payload, synced_at
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            location = EXCLUDED.location,
            source_type = EXCLUDED.source_type,
            headline = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            issued_at = EXCLUDED.issued_at,
            effective_at = EXCLUDED.effective_at,
            payload = EXCLUDED.payload,
            synced_at = EXCLUDED.synced_at;
    """

    records = [
        (
            doc["id"],
            doc["location"],
            doc["source_type"],
            doc["headline"],
            doc["narrative_text"],
            doc["issued_at"],
            doc["effective_at"],
            doc["payload"],
            doc["synced_at"]
        )
        for doc in documents
    ]

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode=sslmode,
        connect_timeout=10
    )
    try:
        with conn.cursor() as cur:
            execute_values(cur, upsert_sql, records, page_size=100)
        conn.commit()
        return len(records)
    finally:
        conn.close()


# Execution Loop
print(f"Processing weather harvesting for locations: {LOCATIONS}\n")

_nws_session = requests.Session()
all_harvested_docs = []

for i, loc in enumerate(LOCATIONS):
    if i > 0:
        time.sleep(1.0)  # Gentle delay between NWS requests
    try:
        docs = fetch_weather_for_location(_nws_session, loc, WEATHER_FETCH_LIMIT)
        print(f"  Harvested {len(docs)} documents for location: {loc}")
        all_harvested_docs.extend(docs)
    except Exception as exc:
        print(f"❌ Skipping '{loc}': harvesting failed ({exc})")
        continue

print(f"\nTotal harvested documents: {len(all_harvested_docs)}")

if all_harvested_docs:
    synced_count = upsert_weather_documents_to_lakebase(all_harvested_docs)
    print(f"✅ Successfully upserted {synced_count} weather documents into '{DOCUMENTS_TABLE_NAME}'.")
else:
    print("ℹ️ No weather documents harvested.")

# COMMAND ----------

# DBTITLE 1,Insert collected news articles using psycopg2
import psycopg2
from psycopg2.extras import execute_values

# Build connection using psycopg2
conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode=sslmode
)

try:
    cursor = conn.cursor()
    
    # Prepare data tuples for batch insert
    insert_data = [
        (
            doc['id'],
            doc['location'],
            doc['source_type'],
            doc['headline'],
            doc['narrative_text'],
            doc['issued_at'],
            doc['effective_at'],
            doc['payload']
        )
        for doc in all_harvested_docs
    ]
    
    # Batch insert into weather documents table with ON CONFLICT clause
    insert_sql = f"""
        INSERT INTO {DOCUMENTS_TABLE_NAME} (
            id, location, source_type, headline, narrative_text,
            issued_at, effective_at, payload, synced_at
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            location = EXCLUDED.location,
            source_type = EXCLUDED.source_type,
            headline = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            issued_at = EXCLUDED.issued_at,
            effective_at = EXCLUDED.effective_at,
            payload = EXCLUDED.payload,
            synced_at = CURRENT_TIMESTAMP
    """
    
    # execute_values provides high-performance batch insertion in psycopg2
    execute_values(
        cursor,
        insert_sql,
        insert_data,
        template="(%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)",
        page_size=100
    )
    
    conn.commit()
    inserted_count = cursor.rowcount
    print(f"✅ Successfully synced {inserted_count} weather documents into {DOCUMENTS_TABLE_NAME}")
    
finally:
    cursor.close()
    conn.close()

print(f"\nReady to compute embeddings! Run the cells below to continue.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load raw weather documents

# COMMAND ----------

import pandas as pd
import psycopg2

# Load weather documents using psycopg2
conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode=sslmode
)

try:
    # Query weather documents with embedding_text constructed from headline & narrative_text
    query = f"""
        SELECT 
            id,
            location,
            source_type,
            headline,
            narrative_text,
            issued_at,
            effective_at,
            TRIM(CONCAT(COALESCE(headline, ''), '. ', COALESCE(narrative_text, ''))) AS embedding_text
        FROM {DOCUMENTS_TABLE_NAME}
        WHERE TRIM(CONCAT(COALESCE(headline, ''), '. ', COALESCE(narrative_text, ''))) IS NOT NULL
          AND TRIM(CONCAT(COALESCE(headline, ''), '. ', COALESCE(narrative_text, ''))) != ''
    """
    
    weather_df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(weather_df)} weather documents from {DOCUMENTS_TABLE_NAME}")
    display(weather_df.head(5))
finally:
    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings

# COMMAND ----------

# DBTITLE 1,Compute embeddings (distributed pandas UDF)
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Set up HuggingFace cache
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# Compute embeddings in batches for memory efficiency
print("Computing embeddings...")
batch_size = 32
all_embeddings = []

for i in range(0, len(weather_df), batch_size):
    batch = weather_df.iloc[i:i+batch_size]
    vectors = model.encode(batch["embedding_text"].tolist(), show_progress_bar=False)
    all_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"   Processed {min(i + batch_size, len(weather_df))}/{len(weather_df)} documents")

# Create embeddings DataFrame
embeddings_df = pd.DataFrame({
    "id": weather_df["id"],
    "location": weather_df["location"],
    "headline": weather_df["headline"],
    "issued_at": weather_df["issued_at"].astype(str),
    "embedding": all_embeddings,
})

print(f"Computed {len(embeddings_df)} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the pgvector destination table exists
# MAGIC

# COMMAND ----------

# Automatically get dimension from the loaded model if not already set
try:
    EMBEDDING_DIM = model.get_sentence_embedding_dimension()
except NameError:
    pass  # Fallback to predefined EMBEDDING_DIM config

# Before running the cells below, ensure you've manually run your database setup script:
#   sql/02_setup_embeddings_table.sql (or sql/02_setup_weather_embeddings_table.sql)
# Replace {{EMBEDDING_DIM}} in that file with the value below:
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {EMBEDDINGS_TABLE_NAME}")
print("\nRun your embeddings SQL table setup in Lakebase before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC

# COMMAND ----------

# DBTITLE 1,Insert embeddings using psycopg2
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Set up HuggingFace cache
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# Compute embeddings in batches
print("Computing embeddings...")
batch_size = 32
all_embeddings = []

for i in range(0, len(weather_df), batch_size):
    batch = weather_df.iloc[i:i+batch_size]
    vectors = model.encode(batch["embedding_text"].tolist(), show_progress_bar=False)
    all_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"   Processed {min(i + batch_size, len(weather_df))}/{len(weather_df)} documents")

# Create embeddings DataFrame matching the weather_embeddings schema
embeddings_df = pd.DataFrame({
    "document_id": weather_df["id"],
    "chunk_index": 0,  # Single chunk per document
    "chunk_text": weather_df["embedding_text"],
    "embedding": all_embeddings,
})

print(f"Computed {len(embeddings_df)} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk weather narrative documents
# MAGIC

# COMMAND ----------

import pandas as pd

# Filter weather_df for valid narrative content
content_df = weather_df[weather_df['narrative_text'].notna() & (weather_df['narrative_text'] != '')].copy()

# Set fallback chunking parameters if not already defined globally
CHUNK_SIZE = globals().get('CHUNK_SIZE', 500)
CHUNK_OVERLAP = globals().get('CHUNK_OVERLAP', 100)

print(f"Chunking narrative text from {len(content_df)} weather documents (CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP})...")

out_doc_ids, out_locations, out_chunk_indexes, out_chunk_texts = [], [], [], []

for idx, row in content_df.iterrows():
    doc_id = row['id']
    location = row['location']
    headline = row.get('headline', '')
    narrative = row['narrative_text']
    
    # Combine headline and narrative text for complete contextual text
    full_text = f"{headline}. {narrative}".strip() if headline else narrative.strip()
    
    if not full_text:
        continue

    # Split into overlapping chunks
    chunk_idx = 0
    for start in range(0, len(full_text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_text = full_text[start : start + CHUNK_SIZE].strip()
        if not chunk_text:
            continue
            
        out_doc_ids.append(doc_id)
        out_locations.append(location)
        out_chunk_indexes.append(chunk_idx)
        out_chunk_texts.append(chunk_text)
        
        chunk_idx += 1
        if start + CHUNK_SIZE >= len(full_text):
            break

chunks_df = pd.DataFrame({
    "document_id": out_doc_ids,
    "location": out_locations,
    "chunk_index": out_chunk_indexes,
    "chunk_text": out_chunk_texts,
})

print(f"Extracted {len(chunks_df)} content chunks across {len(content_df)} weather documents")
display(chunks_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute chunk embeddings

# COMMAND ----------

import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Model should already be loaded from earlier, but ensure cache environment variables are set
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Computing chunk embeddings using {EMBEDDING_MODEL_NAME}...")
# Reuse the model if already loaded, otherwise load it
if 'model' not in locals():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# Compute chunk embeddings in batches
batch_size = 32
all_chunk_embeddings = []

for i in range(0, len(chunks_df), batch_size):
    batch = chunks_df.iloc[i:i+batch_size]
    vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
    all_chunk_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"   Processed {min(i + batch_size, len(chunks_df))}/{len(chunks_df)} chunks")

# Create chunk embeddings DataFrame matching weather_embeddings requirements
chunk_embeddings_df = pd.DataFrame({
    "document_id": chunks_df["document_id"],
    "location": chunks_df["location"],
    "chunk_index": chunks_df["chunk_index"],
    "chunk_text": chunks_df["chunk_text"],
    "embedding": all_chunk_embeddings,
})

print(f"Computed {len(chunk_embeddings_df)} chunk embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the chunk embeddings destination table exists

# COMMAND ----------

# Automatically obtain dimension from the loaded model if defined
try:
    EMBEDDING_DIM = model.get_sentence_embedding_dimension()
except NameError:
    EMBEDDING_DIM = globals().get("EMBEDDING_DIM", 384)

# Set or default the target chunk table name to the weather schema table
CHUNK_EMBEDDINGS_TABLE_NAME = globals().get("CHUNK_EMBEDDINGS_TABLE_NAME", "weather.weather_embeddings")

print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {CHUNK_EMBEDDINGS_TABLE_NAME}")
print(f"\nEnsure `weather.weather_embeddings` table with vector({EMBEDDING_DIM}) is created in your Lakebase database before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert chunk embeddings into Lakebase

# COMMAND ----------

# DBTITLE 1,Insert chunk embeddings using psycopg2
import psycopg2
from psycopg2.extras import execute_values

CHUNK_EMBEDDINGS_TABLE_NAME = globals().get("CHUNK_EMBEDDINGS_TABLE_NAME", "weather.weather_embeddings")
chunk_embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
chunk_embeddings_df['chunk_index'] = chunk_embeddings_df['chunk_index'].astype(int)

chunk_embeddings_rows = chunk_embeddings_df.to_dict('records')

if len(chunk_embeddings_rows) > 0:
    print(f"Inserting {len(chunk_embeddings_rows)} chunk embeddings into {CHUNK_EMBEDDINGS_TABLE_NAME}...")
    
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode=sslmode
    )
    
    try:
        cursor = conn.cursor()
        
        # Changed '{' ... '}' to '[' ... ']' for pgvector format compatibility
        insert_data = [
            (
                row['document_id'],
                int(row['chunk_index']),
                row['chunk_text'],
                '[' + ','.join(str(float(x)) for x in row['embedding']) + ']',
                row['model_name']
            )
            for row in chunk_embeddings_rows
        ]
        
        insert_sql = f"""
            INSERT INTO {CHUNK_EMBEDDINGS_TABLE_NAME} (
                document_id, chunk_index, chunk_text, embedding, model_name
            ) VALUES %s
            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                model_name = EXCLUDED.model_name
        """
        
        template = "(%s, %s, %s, %s::vector, %s)"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)
        
        conn.commit()
        inserted_count = cursor.rowcount
        print(f"✅ Successfully inserted/updated {inserted_count} chunk embeddings into {CHUNK_EMBEDDINGS_TABLE_NAME}")
        
    finally:
        cursor.close()
        conn.close()
else:
    print("No chunk embeddings to write.")
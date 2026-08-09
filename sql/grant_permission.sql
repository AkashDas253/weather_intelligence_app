-- 1. Grant schema usage and creation privileges
GRANT USAGE, CREATE ON SCHEMA weather TO "user";

-- 2. Grant permissions on all existing tables in the schema
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA weather TO "user";

-- 3. Grant permissions on sequences (required for BIGSERIAL primary keys)
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA weather TO "user";

-- 4. Set default privileges for any future tables/sequences created in the weather schema
ALTER DEFAULT PRIVILEGES IN SCHEMA weather 
    GRANT ALL ON TABLES TO "user";

ALTER DEFAULT PRIVILEGES IN SCHEMA weather 
    GRANT ALL ON SEQUENCES TO "user";
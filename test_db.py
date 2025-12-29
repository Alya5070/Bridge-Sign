
import os, psycopg2
url = os.environ.get("SUPABASE_DB_URL")
print("URL set:", bool(url))
if not url:
    raise SystemExit("SUPABASE_DB_URL missing")
conn = psycopg2.connect(url)
with conn.cursor() as cur:
    cur.execute("select count(*) from public.msl_words;")
    print("msl_words rows:", cur.fetchone()[0])
conn.close()


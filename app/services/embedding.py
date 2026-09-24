import os
import psycopg2

from .chunking import ingest_folder


def get_connection_db(): 
    return psycopg2.connect(
        dbname=os.getenv.get("PGDATABASE", "postgres"), 
        username=os.getenv.get("PUSERNAME", "postgres")
    )


import psycopg2
import pandas as pd

def dump_table(
    *,
    table_name: str,
    pg_user: str,
    pg_password: str,
    pg_hostname: str,
    pg_db: str,
    output_path: str,
):
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(host=pg_hostname, database=pg_db, user=pg_user, password=pg_password)
        cursor = conn.cursor()

        table = pd.read_sql(f"SELECT * FROM {table_name};", conn)

        cursor.close()
        conn.close()

    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")

    table.to_parquet(output_path)
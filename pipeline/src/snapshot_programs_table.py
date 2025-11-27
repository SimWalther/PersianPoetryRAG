from utils.dump_table import dump_table
from dotenv import dotenv_values

def main() -> None:
    # Load config
    config = dotenv_values(".env")

    pg_user = config["POSTGRES_USER"]
    pg_password = config["POSTGRES_PASSWORD"]
    pg_hostname = config["POSTGRES_HOSTNAME"]
    pg_db = config["POSTGRES_DB"]

    # Dump embedding table
    dump_table(
        table_name='program_text',
        pg_user=pg_user,
        pg_password=pg_password,
        pg_hostname=pg_hostname,
        pg_db=pg_db,
        output_path='data/raw/programs.parquet'
    )

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# scripts/migrate.py
"""
Run all pending database migrations.

Usage:
    python scripts/migrate.py
    python scripts/migrate.py --db path/to/custom.db
"""
import argparse
import sys
from src.db.sqlite import run_migrations


def main():
    parser = argparse.ArgumentParser(description="Run pending database migrations.")
    parser.add_argument("--db", default=None, help="Path to SQLite database file (uses SQLITE_DB_PATH from .env by default)")
    args = parser.parse_args()

    print("Running migrations...")
    run_migrations(db_path=args.db)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())

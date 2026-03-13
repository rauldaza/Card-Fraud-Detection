"""
mlflow_server.py — Start a local MLflow tracking server.

Downloads the single shared mlflow.db from S3 (contains ALL runs from
both training and evaluation jobs), then starts the UI locally.

Usage:
    python mlflow_server.py              # sync from S3 then start
    python mlflow_server.py --no-sync    # skip sync (faster restart)
    python mlflow_server.py --port 5001  # custom port
"""

import os
import argparse
import subprocess

from utils.mlflow_utils import download_mlflow_db, MLFLOW_S3_BUCKET, MLFLOW_S3_KEY

# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_HOST      = "127.0.0.1"
DEFAULT_PORT      = 5000
DEFAULT_DB_PATH   = "./mlflow.db"
DEFAULT_ARTIFACTS = f"s3://{MLFLOW_S3_BUCKET}/mlflow/artifacts"


# ─── Server ──────────────────────────────────────────────────────────────────

def start_server(host: str, port: int, db_path: str, artifacts: str, skip_sync: bool):
    if not skip_sync:
        print("Syncing mlflow.db from S3...")
        download_mlflow_db(local_path=db_path)
    else:
        print("Skipping S3 sync (--no-sync)")

    cmd = [
        "mlflow", "server",
        "--host",                  host,
        "--port",                  str(port),
        "--backend-store-uri",     f"sqlite:///{os.path.abspath(db_path)}",
        "--default-artifact-root", artifacts,
    ]

    print(f"\nStarting MLflow server...")
    print(f"  URL       : http://{host}:{port}")
    print(f"  DB        : {db_path}")
    print(f"  Artifacts : {artifacts}")
    print(f"\n  Open your browser → http://{host}:{port}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nMLflow server stopped.")
    except Exception as e:
        print(f"Error starting MLflow server: {e}")
        raise


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a local MLflow tracking server.")
    parser.add_argument("--host",      default=DEFAULT_HOST)
    parser.add_argument("--port",      type=int, default=DEFAULT_PORT)
    parser.add_argument("--db-path",   default=DEFAULT_DB_PATH,
                        help="Local path for the downloaded mlflow.db")
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS,
                        help="Artifact root URI (default: S3)")
    parser.add_argument("--no-sync",   action="store_true",
                        help="Skip downloading mlflow.db from S3")
    args = parser.parse_args()

    start_server(
        host      = args.host,
        port      = args.port,
        db_path   = args.db_path,
        artifacts = args.artifacts,
        skip_sync = args.no_sync,
    )
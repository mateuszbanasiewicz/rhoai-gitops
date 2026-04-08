#!/usr/bin/env python3
"""
Pobiera model z Hugging Face i uploaduje go do S3 pod ścieżką:
  s3://<bucket>/huggingface/<nazwa_modelu>/
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import boto3
from huggingface_hub import snapshot_download
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_model(model_id: str, cache_dir: Path) -> Path:
    """Pobiera wszystkie pliki modelu do lokalnego cache."""
    log.info(f"Pobieranie modelu '{model_id}' z Hugging Face...")
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        local_dir=str(cache_dir / model_id.replace("/", "--")),
        local_dir_use_symlinks=False,
    )
    log.info(f"Model zapisany lokalnie w: {local_dir}")
    return Path(local_dir)


def upload_to_s3(
    local_path: Path,
    bucket: str,
    model_id: str,
    s3_client,
    endpoint_url: Optional[str] = None,
) -> None:
    """
    Rekurencyjnie uploaduje zawartość local_path do:
      s3://<bucket>/huggingface/<model_id>/
    """
    # Nazwa modelu bez prefixu organizacji do ścieżki S3 – zachowaj pełną nazwę
    s3_prefix = f"huggingface/{model_id}"

    files = list(local_path.rglob("*"))
    total = sum(1 for f in files if f.is_file())
    log.info(f"Uploading {total} plików do s3://{bucket}/{s3_prefix}/")

    uploaded = 0
    for file in files:
        if not file.is_file():
            continue

        relative = file.relative_to(local_path)
        s3_key = f"{s3_prefix}/{relative}"

        log.info(f"  [{uploaded + 1}/{total}] {relative} -> s3://{bucket}/{s3_key}")
        s3_client.upload_file(str(file), bucket, s3_key)
        uploaded += 1

    log.info(f"Upload zakończony. Przesłano {uploaded} plików.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pobiera model z Hugging Face i zapisuje go na S3."
    )
    p.add_argument(
        "--model-id",
        required=True,
        help="ID modelu na HF, np. 'microsoft/phi-2' lub 'bert-base-uncased'",
    )
    p.add_argument(
        "--bucket",
        required=True,
        help="Nazwa bucketu S3",
    )
    p.add_argument(
        "--cache-dir",
        default="/tmp/hf_cache",
        help="Lokalny katalog cache (domyślnie: /tmp/hf_cache)",
    )
    p.add_argument(
        "--endpoint-url",
        default=None,
        help="Opcjonalny endpoint S3 (np. dla MinIO / ODF / Ceph: http://minio.example.com:9000)",
    )
    p.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Token Hugging Face dla prywatnych modeli (lub zmienna HF_TOKEN)",
    )
    p.add_argument(
        "--aws-access-key",
        default=os.getenv("AWS_ACCESS_KEY_ID"),
        help="AWS Access Key ID (lub zmienna AWS_ACCESS_KEY_ID)",
    )
    p.add_argument(
        "--aws-secret-key",
        default=os.getenv("AWS_SECRET_ACCESS_KEY"),
        help="AWS Secret Access Key (lub zmienna AWS_SECRET_ACCESS_KEY)",
    )
    p.add_argument(
        "--aws-region",
        default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        help="Region AWS (domyślnie: us-east-1)",
    )
    p.add_argument(
        "--keep-local",
        action="store_true",
        help="Nie usuwaj lokalnych plików po uploadzie",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Ustaw token HF jeśli podany
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Krok 1: pobierz model
    local_model_path = download_model(args.model_id, cache_dir)

    # Krok 2: utwórz klienta S3
    session_kwargs: dict = {}
    if args.aws_access_key and args.aws_secret_key:
        session_kwargs["aws_access_key_id"] = args.aws_access_key
        session_kwargs["aws_secret_access_key"] = args.aws_secret_key

    client_kwargs: dict = {"region_name": args.aws_region}
    if args.endpoint_url:
        client_kwargs["endpoint_url"] = args.endpoint_url

    s3 = boto3.client("s3", **session_kwargs, **client_kwargs)

    # Krok 3: upload do S3
    upload_to_s3(
        local_path=local_model_path,
        bucket=args.bucket,
        model_id=args.model_id,
        s3_client=s3,
        endpoint_url=args.endpoint_url,
    )

    # Krok 4: opcjonalne czyszczenie
    if not args.keep_local:
        import shutil
        log.info(f"Usuwanie lokalnego cache: {local_model_path}")
        shutil.rmtree(local_model_path, ignore_errors=True)

    log.info("Gotowe!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Przerwano przez użytkownika.")
        sys.exit(1)
    except Exception as exc:
        log.error(f"Błąd: {exc}", exc_info=True)
        sys.exit(1)
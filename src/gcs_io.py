"""Manifests, GCS upload/download of data bundles, and the run receipt. Credentials come from Application Default Credentials.

A bundle is a folder with a manifest.json that lists every other file with its size, SHA-256 and (for .jsonl) row count.
A run is complete when its receipt exists at <data_uri>/receipts/<run_id>.json. The receipt is written last, after the
raw, prepared and images bundles, so Notebook 2 only has to look for the newest receipt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

KINDS = ("raw", "prepared", "images")
RECEIPT_SCHEMA = "safeshare-data-receipt-v3"


# ------------------------------------------------------------------ local manifests

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def file_entries(root):
    """Size, SHA-256 and (for .jsonl) row count of every file under root except the root manifest."""
    root, entries = Path(root), {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative != "manifest.json":
            entry = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            if path.suffix == ".jsonl":
                entry["rows"] = count_rows(path)
            entries[relative] = entry
    return entries


def write_manifest(root, fields, release_prefix):
    """Write root/manifest.json: the given fields, every file's hash, and a release_id derived from them."""
    manifest = {**fields, "files": file_entries(root)}
    manifest["release_id"] = release_prefix + hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    (Path(root) / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_bundle(directory):
    """Check every file listed in manifest.json (size, SHA-256, row count). Returns the manifest."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("files"):
        raise ValueError("The manifest lists no files.")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if (not path.is_file() or path.stat().st_size != expected["size_bytes"]
                or sha256_file(path) != expected["sha256"]):
            raise ValueError(f"Missing or changed file: {relative}")
        if "rows" in expected and count_rows(path) != expected["rows"]:
            raise ValueError(f"Row count changed: {relative}")
    return manifest


def save_raw_snapshot(source, config, directory):
    """Save the unchanged publisher records as Parquet, with attribution and a manifest."""
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new empty directory for the raw snapshot.")
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "test"):
        source[split].to_parquet(str(root / f"{split}.parquet"))
        counts[split] = len(source[split])
    (root / "ATTRIBUTION.txt").write_text(
        "Nemotron-PII, NVIDIA Corporation (2025).\n"
        "Authors: Amy Steier, Andre Manoel, Alexa Haushalter, Maarten Van Segbroeck.\n"
        "Source: https://huggingface.co/datasets/nvidia/Nemotron-PII\n"
        "License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\n"
        "This raw snapshot retains source values and split order; files were re-encoded as Parquet.\n"
        "No relationship, endorsement, or production privacy guarantee is implied.\n", encoding="utf-8")
    manifest = write_manifest(root, {"schema_version": "safeshare-raw-v1", "kind": "raw_source_snapshot",
                                     "dataset_id": config["dataset_id"], "dataset_revision": config["dataset_revision"],
                                     "source_counts": counts}, "safeshare-raw-")
    verify_bundle(root)
    return manifest


# ------------------------------------------------------------------ GCS

def parse_gcs_uri(uri, expected_bucket=None):
    """Split a full gs://bucket/prefix location into (bucket, prefix)."""
    bucket, _, prefix = uri[5:].partition("/") if isinstance(uri, str) and uri.startswith("gs://") else ("", "", "")
    prefix = prefix.strip("/")
    if not bucket or not prefix:
        raise ValueError("Use a full GCS prefix, for example gs://bucket/safeshare/data.")
    if expected_bucket and bucket != expected_bucket:
        raise ValueError("The bucket in the GCS location differs from GCS_BUCKET_NAME.")
    return bucket, prefix


def _client(project_id, client):
    if client is not None:
        return client
    from google.cloud import storage
    return storage.Client(project=project_id)


def upload_bundle(directory, destination_uri, project_id, expected_bucket=None, client=None):
    """Upload a bundle to a NEW GCS prefix. Existing objects are never overwritten.

    The client library checks each file's checksum during upload, and the stored size is compared with the
    manifest. manifest.json goes last, so a prefix that has a manifest is a complete upload. If an upload
    fails partway, use a new run ID.
    """
    root = Path(directory)
    manifest = verify_bundle(root)
    bucket_name, prefix = parse_gcs_uri(destination_uri, expected_bucket)
    bucket = _client(project_id, client).bucket(bucket_name)
    bucket.reload()                                    # fails clearly if the bucket is not accessible
    if bucket.blob(f"{prefix}/manifest.json").exists():
        raise FileExistsError("This GCS version already has a manifest. Use a new run ID.")
    for relative, expected in manifest["files"].items():
        blob = bucket.blob(f"{prefix}/{relative}")
        blob.upload_from_filename(str(root / relative), if_generation_match=0, checksum="auto")
        if blob.size != expected["size_bytes"]:        # blob.size comes from the upload response
            raise OSError(f"Stored size differs for {relative}. The manifest was not published.")
    bucket.blob(f"{prefix}/manifest.json").upload_from_filename(
        str(root / "manifest.json"), if_generation_match=0, content_type="application/json", checksum="auto")
    return {"gcs_uri": f"gs://{bucket_name}/{prefix}", "manifest_sha256": sha256_file(root / "manifest.json"),
            "verified_files": len(manifest["files"])}


def download_bundle(source_uri, directory, project_id, expected_manifest_sha256, expected_bucket=None, client=None,
                    only=None):
    """Download one completed bundle (or only the files whose paths start with a prefix in `only`).

    The manifest hash from the receipt is the trust anchor; every downloaded file is checked against it.
    """
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new empty directory for the download.")
    root.mkdir(parents=True, exist_ok=True)
    bucket_name, prefix = parse_gcs_uri(source_uri, expected_bucket)
    bucket = _client(project_id, client).bucket(bucket_name)
    bucket.blob(f"{prefix}/manifest.json").download_to_filename(str(root / "manifest.json"), checksum="auto")
    if sha256_file(root / "manifest.json") != expected_manifest_sha256:
        raise OSError("The downloaded manifest does not match the hash from the receipt.")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    wanted = [r for r in manifest["files"] if only is None or any(r.startswith(p) for p in only)]
    if not wanted:
        raise ValueError(f"No files in the bundle match {only}.")
    for relative in wanted:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe path in manifest: {relative}")
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(f"{prefix}/{relative}").download_to_filename(str(root / relative), checksum="auto")
        expected, path = manifest["files"][relative], root / relative
        if path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]:
            raise OSError(f"Downloaded file does not match the manifest: {relative}")
        if "rows" in expected and count_rows(path) != expected["rows"]:
            raise OSError(f"Row count does not match the manifest: {relative}")
    return {"directory": str(root), "verified_files": len(wanted), "manifest": manifest}


# ------------------------------------------------------------------ receipts

def write_receipt(receipt, data_uri, project_id, expected_bucket=None, client=None):
    """Publish the run receipt at <data_uri>/receipts/<run_id>.json. Call this after every bundle is uploaded."""
    missing = [key for key in ("run_id", *KINDS) if key not in receipt]
    if missing:
        raise ValueError(f"The receipt is missing: {', '.join(missing)}")
    receipt = {"schema_version": RECEIPT_SCHEMA, "status": "complete", **receipt}
    bucket_name, prefix = parse_gcs_uri(data_uri, expected_bucket)
    blob = _client(project_id, client).bucket(bucket_name).blob(f"{prefix}/receipts/{receipt['run_id']}.json")
    blob.upload_from_string(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            content_type="application/json", if_generation_match=0)
    return f"gs://{bucket_name}/{blob.name}"


def find_latest_receipt(data_uri, project_id, run_id=None, expected_bucket=None, client=None):
    """Return the newest receipt (or the one for run_id) after checking that each bundle's manifest matches it.

    Run IDs start with a UTC timestamp, so the highest name is the latest run. Only the manifests are
    downloaded here; download_bundle checks every file.
    """
    bucket_name, prefix = parse_gcs_uri(data_uri, expected_bucket)
    client = _client(project_id, client)
    bucket = client.bucket(bucket_name)
    if run_id:
        name = f"{prefix}/receipts/{run_id}.json"
    else:
        names = sorted(blob.name for blob in client.list_blobs(bucket_name, prefix=f"{prefix}/receipts/")
                       if blob.name.endswith(".json"))
        if not names:
            raise FileNotFoundError(f"No receipts under gs://{bucket_name}/{prefix}/receipts/. Run Notebook 1 first.")
        name = names[-1]
    receipt = json.loads(bucket.blob(name).download_as_bytes())
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "complete":
        raise ValueError(f"{name} is not a complete {RECEIPT_SCHEMA} receipt.")
    if Path(name).stem != receipt.get("run_id"):
        raise ValueError(f"{name} holds a receipt for a different run.")
    for kind in KINDS:
        bundle_bucket, bundle_prefix = parse_gcs_uri(receipt[kind]["gcs_uri"], expected_bucket)
        manifest = client.bucket(bundle_bucket).blob(f"{bundle_prefix}/manifest.json").download_as_bytes()
        if hashlib.sha256(manifest).hexdigest() != receipt[kind]["manifest_sha256"]:
            raise ValueError(f"The {kind} manifest in GCS does not match the receipt.")
    return receipt

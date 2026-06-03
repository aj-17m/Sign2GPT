"""
Reset (delete) all ISL upload data on the network volume.

Run from your laptop or any computer with Python + internet access.
Doesn't require a RunPod pod. Uses the S3 API to delete files directly.

WHAT IT DELETES:
  - All uploaded videos under s3://<bucket>/submissions/
  - The SQLite database at s3://<bucket>/_state/isl_data.db
  - (Optionally with --all) any training artifacts that may be on the volume

USAGE:

    # Install dependency (one-time)
    pip install boto3

    # Standard reset - removes ALL submissions + database
    python reset_isl_data.py \\
        --bucket p014akuq8i \\
        --endpoint https://s3api-eu-ro-1.runpod.io \\
        --region eu-ro-1 \\
        --access-key user_xxxxx \\
        --secret-key rps_xxxxx

    # Or use environment variables (cleaner):
    export S3_BUCKET=p014akuq8i
    export S3_ENDPOINT_URL=https://s3api-eu-ro-1.runpod.io
    export S3_REGION=eu-ro-1
    export S3_ACCESS_KEY=user_xxxxx
    export S3_SECRET_KEY=rps_xxxxx

    python reset_isl_data.py

    # Dry-run (show what would be deleted without actually deleting)
    python reset_isl_data.py --dry-run

    # Skip confirmation prompt (dangerous - for scripts/CI)
    python reset_isl_data.py --yes

    # Also delete training artifacts (frames, LMDB, checkpoints)
    python reset_isl_data.py --all

SAFETY:
  - Lists what will be deleted before doing anything
  - Asks you to type "DELETE" to confirm (unless --yes)
  - Has --dry-run mode
"""

import argparse
import os
import sys

try:
    import boto3
except ImportError:
    print("[err] boto3 not installed. Run: pip install boto3", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Reset ISL upload data on the network volume")
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET"), help="Bucket / volume ID (env: S3_BUCKET)")
    p.add_argument("--endpoint", default=os.environ.get("S3_ENDPOINT_URL"), help="Endpoint URL (env: S3_ENDPOINT_URL)")
    p.add_argument("--region", default=os.environ.get("S3_REGION", "us-east-1"), help="Region (env: S3_REGION)")
    p.add_argument("--access-key", default=os.environ.get("S3_ACCESS_KEY"), help="Access key (env: S3_ACCESS_KEY)")
    p.add_argument("--secret-key", default=os.environ.get("S3_SECRET_KEY"), help="Secret key (env: S3_SECRET_KEY)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be deleted without doing it")
    p.add_argument("--yes", action="store_true", help="Skip the typed confirmation prompt")
    p.add_argument("--all", action="store_true",
                   help="Also delete training artifacts (frames, LMDB, checkpoints, logs)")
    return p.parse_args()


def list_objects(s3, bucket, prefix):
    """Return list of all keys under a prefix."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def delete_keys(s3, bucket, keys, dry_run):
    """Delete keys in batches of 1000 (S3 API limit)."""
    n_deleted = 0
    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        if dry_run:
            n_deleted += len(batch)
            continue
        resp = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch]},
        )
        n_deleted += len(resp.get("Deleted", []))
    return n_deleted


def main():
    args = parse_args()

    # Validate required values
    missing = []
    if not args.bucket: missing.append("--bucket / S3_BUCKET")
    if not args.access_key: missing.append("--access-key / S3_ACCESS_KEY")
    if not args.secret_key: missing.append("--secret-key / S3_SECRET_KEY")
    if missing:
        print(f"[err] Missing required: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # Set up S3 client
    s3_kwargs = {
        "aws_access_key_id": args.access_key,
        "aws_secret_access_key": args.secret_key,
        "region_name": args.region,
    }
    if args.endpoint:
        s3_kwargs["endpoint_url"] = args.endpoint
    s3 = boto3.client("s3", **s3_kwargs)

    # Build list of prefixes to delete
    prefixes = [
        ("submissions/", "uploaded videos"),
        ("_state/", "database (mapping file)"),
    ]
    if args.all:
        prefixes.extend([
            # These prefixes are used by training but a pure S3 reset wouldn't
            # have created them. They exist only if someone ran training and
            # the results were synced/written to the volume. Listing them is
            # safe; deletion is no-op if they don't exist.
            ("data/isl/", "training frames + annotations.csv"),
            ("lmdb/isl/", "LMDB cache"),
            ("checkpoints/isl_stage1_config/", "trained stage 1 checkpoints"),
            ("checkpoints/isl_stage2_config/", "trained stage 2 checkpoints"),
            ("results/", "training logs"),
        ])

    # Enumerate everything first
    print(f"Connecting to s3://{args.bucket} ...")
    plan = []
    total_size = 0
    for prefix, desc in prefixes:
        keys = list_objects(s3, args.bucket, prefix)
        if not keys:
            continue
        # Get total size for this prefix
        size = 0
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=args.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    size += obj.get("Size", 0)
        except Exception:
            pass
        plan.append((prefix, desc, keys, size))
        total_size += size

    if not plan:
        print("[ok] Nothing to delete - bucket is already empty for these prefixes.")
        return

    # Show plan
    print("\n" + "=" * 70)
    print("  RESET PLAN")
    print("=" * 70)
    for prefix, desc, keys, size in plan:
        size_mb = size / (1024 * 1024)
        print(f"  {prefix:50s} {len(keys):5d} files  ({size_mb:.1f} MB) - {desc}")
    total_mb = total_size / (1024 * 1024)
    total_files = sum(len(k) for _, _, k, _ in plan)
    print(f"  {'TOTAL':50s} {total_files:5d} files  ({total_mb:.1f} MB)")
    print("=" * 70)

    if args.dry_run:
        print("\n[dry-run] No files were actually deleted. Remove --dry-run to delete.")
        return

    # Confirm
    if not args.yes:
        print("\n⚠️  This will PERMANENTLY DELETE these files. Cannot be undone.")
        print("    (Back them up first with `aws s3 sync` if you want to keep them.)")
        confirm = input('\nType "DELETE" (all caps) to confirm: ')
        if confirm != "DELETE":
            print("[cancel] You didn't type DELETE. Aborting, no files touched.")
            sys.exit(1)

    # Delete
    print()
    grand_total = 0
    for prefix, desc, keys, _ in plan:
        print(f"Deleting {len(keys)} files under {prefix} ...")
        n = delete_keys(s3, args.bucket, keys, dry_run=False)
        print(f"  ✓ {n} deleted")
        grand_total += n

    print(f"\n[done] Deleted {grand_total} files in total.")
    print(f"       The bucket s3://{args.bucket}/ is now reset.")
    print(f"       Web app will create a fresh database on the next upload.")


if __name__ == "__main__":
    main()

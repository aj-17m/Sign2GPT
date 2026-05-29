"""
Map the model's printed predictions in stage2.log back to their clip names.

The trainer's autoreg eval prints lines like:
    0 ABLEU: <prediction>
    0   TGT: <ground truth>
But doesn't include the clip name. We can recover the name by walking the
dev (or test) set in the same order the trainer's dataloader did.

Dataloader order is deterministic:
  - valid_ds_params has shuffle=False
  - get_ds() does `df.groupby("name")` -> alphabetical clip-name order
  - PHX clips with no LMDB are filtered out (our b18c857 patch)

So prediction N in the log corresponds to clip-N in the
alphabetically-sorted, LMDB-filtered split CSV.

Usage:
    python scripts/match_predictions_to_clips.py \\
        --split dev \\
        --log /workspace/results/stage2.log \\
        --out /workspace/dev_clip_predictions.csv

Then open /workspace/dev_clip_predictions.csv to find any clip's prediction.
"""

import argparse
import os
import re

import pandas as pd

REPO_ROOT = "/workspace/Sign2GPT"
CSV_DIR = f"{REPO_ROOT}/data/phoenix2014t"
LMDB_ROOT = "/workspace/lmdb/phoenix2014t/lmdb_videos"


def parse_log_samples(log_path):
    """Yield (idx_within_batch, ableu_text, tgt_text) tuples from the log."""
    samples = []
    with open(log_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        m = re.match(r"^(\d+)\s+ABLEU:\s*(.*)$", lines[i].rstrip())
        if m:
            idx = int(m.group(1))
            pred = m.group(2).strip()
            tgt = None
            # The TGT line is the next non-separator line
            j = i + 1
            while j < len(lines):
                tm = re.match(r"^\d+\s+TGT:\s*(.*)$", lines[j].rstrip())
                if tm:
                    tgt = tm.group(1).strip()
                    break
                if "ABLEU:" in lines[j]:
                    break
                j += 1
            samples.append((idx, pred, tgt))
            i = j + 1
        else:
            i += 1
    return samples


def get_split_clips_in_dataloader_order(split):
    """Return clip names in the same order the dataloader processed them."""
    csv_path = f"{CSV_DIR}/PHOENIX-2014-T.{split}.corpus.csv"
    df = pd.read_csv(csv_path, sep="|")

    # Apply the same filter as our dataloader patch (b18c857)
    df = df[df["name"].apply(lambda n: os.path.isdir(f"{LMDB_ROOT}/{n}"))]

    # df.groupby("name") iterates groups in sorted order (default behavior)
    sorted_names = sorted(df["name"].unique().tolist())
    return sorted_names, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--log", required=True, help="Path to stage2.log")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    print(f"[match] Parsing samples from {args.log}...")
    samples = parse_log_samples(args.log)
    print(f"[match] Found {len(samples)} (ABLEU, TGT) pairs in the log")

    print(f"[match] Loading {args.split} split clip names...")
    clip_names, df = get_split_clips_in_dataloader_order(args.split)
    print(f"[match] {len(clip_names)} clips in {args.split} after LMDB filter")

    # The log may contain samples from multiple evaluation passes (e.g. one
    # validation per epoch). The final pass should have len(clip_names) samples
    # (rounded up by batch). Take the LAST len(clip_names) samples.
    final_samples = samples[-len(clip_names):] if len(samples) > len(clip_names) else samples
    print(f"[match] Using last {len(final_samples)} samples (final eval epoch)")

    if len(final_samples) < len(clip_names):
        print(f"[match] WARNING: only {len(final_samples)} samples but {len(clip_names)} clips - alignment may be off")

    # Build the mapping
    rows = []
    df_indexed = df.set_index("name")
    for i, clip_name in enumerate(clip_names):
        if i >= len(final_samples):
            break
        _, pred, tgt = final_samples[i]
        gt_from_csv = df_indexed.loc[clip_name, "translation"] if clip_name in df_indexed.index else "?"

        # Sanity check: does the TGT in the log match the CSV?
        match = "OK" if (tgt and gt_from_csv and tgt.strip().lower() in gt_from_csv.strip().lower()) or (
            tgt and gt_from_csv and gt_from_csv.strip().lower() in tgt.strip().lower()
        ) else "MISMATCH"

        rows.append({
            "order": i,
            "clip_name": clip_name,
            "split": args.split,
            "prediction": pred,
            "ground_truth_from_log": tgt,
            "ground_truth_from_csv": gt_from_csv,
            "sanity_check": match,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, sep="|", index=False)

    # Print summary
    n_ok = (out_df["sanity_check"] == "OK").sum()
    n_mismatch = (out_df["sanity_check"] == "MISMATCH").sum()
    print(f"\n[match] Written {len(rows)} clip-prediction pairs to {args.out}")
    print(f"[match] Sanity check: {n_ok} OK, {n_mismatch} mismatched")

    if n_mismatch > len(rows) * 0.3:
        print("[match] WARNING: many mismatches - alignment is likely WRONG.")
        print("       Possible reasons: shuffling on eval, multiple eval passes mixed, etc.")
    else:
        print("[match] Alignment looks correct.")

    # Show a few examples
    print("\n[match] First 5 paired translations:")
    print("=" * 70)
    for _, row in out_df.head(5).iterrows():
        print(f"Clip:       {row['clip_name']}")
        print(f"Prediction: {row['prediction']}")
        print(f"Ground truth: {row['ground_truth_from_csv']}")
        print(f"Sanity: {row['sanity_check']}")
        print("-" * 70)


if __name__ == "__main__":
    main()

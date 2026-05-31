"""
English version of pseudo_gloss_de.py.

Reads ISL split CSVs, lemmatizes English translations with spaCy
(en_core_web_lg), keeps content words (nouns, verbs, adjectives, etc.),
and produces a pkl file with the same structure as the German one:

    {
        "dict_lem_counter": {lemma: count, ...},
        "dict_sentence":    {sentence: [lemma, lemma, ...], ...},
        "dict_lem_to_id":   {lemma: int_id, ...},
    }

Usage:
    python scripts/pseudo_gloss_en.py \\
        --csv_dir /workspace/Sign2GPT/data/isl \\
        --output_pkl /workspace/Sign2GPT/data/isl/processed_words.isl_pkl

Prerequisites:
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
    # FastText will auto-download cc.en.300.bin (~7 GB) on first run
"""

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import spacy
import fasttext.util
from tqdm import tqdm


# Content word POS tags we keep (skip determiners, prepositions, etc.)
SELECTED_VOCAB = {"NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"}


def get_parts_of_speech(nlp, sentence):
    """Return (lemma_lower, token_text, pos_tag) for each token."""
    doc = nlp(sentence)
    return [(token.lemma_.lower(), token.text, token.pos_) for token in doc]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", required=True,
                        help="Directory with ISL.{train,dev,test}.corpus.csv files")
    parser.add_argument("--output_pkl", required=True,
                        help="Output pkl path (e.g., processed_words.isl_pkl)")
    parser.add_argument("--csv_pattern", default="ISL.{}.corpus.csv",
                        help="CSV filename pattern (default: ISL.{train,dev,test}.corpus.csv)")
    parser.add_argument("--text_col", default="translation",
                        help="Column name containing the text (default: translation)")
    parser.add_argument("--skip_fasttext", action="store_true",
                        help="Skip FastText download (faster, but the pkl won't have embeddings)")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_path = Path(args.output_pkl)

    # Load spaCy English model
    print("[pseudo_gloss_en] Loading spaCy en_core_web_lg...")
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        print("[err] en_core_web_lg not found. Install with:", file=sys.stderr)
        print("      pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl",
              file=sys.stderr)
        sys.exit(1)

    # Read all 3 splits and combine
    dfs = []
    for split in ["train", "dev", "test"]:
        csv_path = csv_dir / args.csv_pattern.format(split)
        if not csv_path.is_file():
            print(f"[warn] CSV not found: {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path, sep="|")
        dfs.append(df)

    if not dfs:
        print(f"[err] No CSV files found in {csv_dir}", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"[pseudo_gloss_en] Total sentences across splits: {len(df)}")

    if args.text_col not in df.columns:
        print(f"[err] Column '{args.text_col}' not found. Available: {list(df.columns)}",
              file=sys.stderr)
        sys.exit(1)

    sentences = df[args.text_col].astype(str).values

    # Lemmatize each sentence, keep only selected POS tags
    dict_sentence = {}
    all_lems = []
    for sentence in tqdm(sentences, desc="Lemmatizing"):
        pos = get_parts_of_speech(nlp, sentence)
        lems = []
        for lem, _word, pos_tag in pos:
            if pos_tag in SELECTED_VOCAB:
                lems.append(lem)
        all_lems.extend(lems)
        dict_sentence[sentence] = lems

    dict_lem_to_id = {l: i for i, l in enumerate(sorted(set(all_lems)))}
    print(f"[pseudo_gloss_en] Unique lemmas (vocab size): {len(dict_lem_to_id)}")
    print(f"[pseudo_gloss_en] Most common lemmas:")
    for lem, count in Counter(all_lems).most_common(20):
        print(f"    {lem:20s} {count}")

    # Optional: download FastText English (only if not skipped)
    if not args.skip_fasttext:
        print("[pseudo_gloss_en] Downloading FastText cc.en.300.bin (~7 GB if not cached)...")
        try:
            fasttext.util.download_model("en", if_exists="ignore")
            ft = fasttext.load_model("cc.en.300.bin")
            print(f"[pseudo_gloss_en] FastText loaded (dim={ft.get_dimension()})")
        except Exception as e:
            print(f"[warn] FastText download/load failed: {e}", file=sys.stderr)
            print("       Continuing without FastText. The model can still train.", file=sys.stderr)

    # Build the output dict (same structure as German version)
    dict_processed_words = {
        "dict_lem_counter": dict(Counter(all_lems)),
        "dict_sentence": dict(dict_sentence),
        "dict_lem_to_id": dict_lem_to_id,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dict_processed_words, f)

    print(f"\n[pseudo_gloss_en] Wrote {out_path} ({len(dict_lem_to_id)} classes)")
    print(f"[pseudo_gloss_en] Use this in your ISL config:")
    print(f"    cfg.train_ds_params.pseudo_gloss_dir = '{out_path}'")
    print(f"    post_params['emb_pkl_dir'] = '{out_path}'")
    print(f"    post_params['emb_lang'] = 'en'")


if __name__ == "__main__":
    main()

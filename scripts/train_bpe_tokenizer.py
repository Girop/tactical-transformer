#!/usr/bin/env python
"""Train a byte-level BPE tokenizer on the SMT-LIB corpus and save it in
a directory loadable via `AutoTokenizer.from_pretrained(out_dir)`.

Usage: venv/bin/python scripts/train_bpe_tokenizer.py [--smtlib-root DIR] [--out-dir DIR] [--vocab-size N]
"""
import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smtlib-root", type=Path, default=Path("./smtlib/"))
    parser.add_argument("--out-dir", type=Path, default=Path("./tokenizer/smtlib-bpe"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    files = [str(p) for p in args.smtlib_root.rglob("*.smt2")]
    if len(files) == 0:
        raise SystemExit(f"No .smt2 files found under {args.smtlib_root}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files, vocab_size=args.vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = str(args.out_dir / "tokenizer.json")
    tokenizer.save(out_json)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(out_json),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>"
    )
    fast_tokenizer.save_pretrained(str(args.out_dir))
    print(f"Trained on {len(files)} files. Saved {len(fast_tokenizer)}-token vocab to {args.out_dir}")


if __name__ == "__main__":
    main()

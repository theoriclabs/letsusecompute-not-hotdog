# Not Hotdog on compute.cx

Train a tiny CNN that tells hot dogs from everything else — the SeeFood gag from *Silicon Valley* — on a fresh cloud GPU via [compute.cx](https://compute.cx).

Guide: https://letsusecompute.com/posts/not-hotdog

## Quick start

```bash
curl -fsSL https://compute.cx/install.sh | sh
compute setup
compute credits add 10

compute run train.py::train --gpu cheap --provider vastai --dry-run
compute run train.py::train --gpu cheap --provider vastai --timeout 1200 --wait
```

`--dry-run` only prints the upload plan. Cost and GPU show up on the real run, in the preflight quote, before you confirm spend.

After the run succeeds and the machine is gone:

```bash
compute artifacts list <run_id>
compute artifacts get <run_id> <artifact_id> <version> --out ./weights
```

## What this trains

- Randomly initialized CNN (~180k params): 3 conv blocks → adaptive pool → linear
- Loss: cross-entropy
- Default: 5 epochs, 128×128 images
- Dataset: [`theoriclabs/hot-dog-not-hot-dog`](https://huggingface.co/datasets/theoriclabs/hot-dog-not-hot-dog) (Food-101 binary cut; see dataset card)

## Agent path

Point your agent at [`SKILL.md`](./SKILL.md) and ask it to train Not Hotdog on compute.

## Caps to know

- One active run per account
- New accounts: first-day spend cap (~$50)
- Balance ≤ $1 blocks new runs
- Weights via artifacts are kept for a limited window; we also publish this guide’s checkpoint on Hugging Face

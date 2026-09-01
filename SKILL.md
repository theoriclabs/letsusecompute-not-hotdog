---
name: not-hotdog-compute
description: >-
  Train the Not Hotdog CNN on compute.cx. Use when the user wants to run the
  letsusecompute Not Hotdog guide, train a hot-dog classifier on a cloud GPU,
  dry-run the payload, get a spend quote, or download trained weights.
---

# Not Hotdog on compute.cx

Train a randomly initialized CNN (3 conv blocks, ~180k params, cross-entropy)
on the binary hot-dog dataset. One fresh GPU per run. Weights come back via
`compute artifacts get`.

Dataset: `theoriclabs/hot-dog-not-hot-dog`  
Script: `train.py::train` in this repo  
Human guide: https://letsusecompute.com/posts/not-hotdog

## Rules

- GPU runs spend prepaid credit. Confirm the quote.
- `--dry-run` prints the AST upload plan only. **No quote, no machine.** To see GPU, time, and dollar estimate, run without `--dry-run` and read the preflight (cancel if you do not want to spend).
- Prefer `--gpu cheap --provider vastai` for this tiny CNN. Do not default to MI300X.
- Do not invent files. Use `train.py` from this repo.
- Do not paste API keys or secret values into chat.

## Steps

1. Install and sign in:

```bash
curl -fsSL https://compute.cx/install.sh | sh
compute setup
compute credits add 10
```

2. Dry-run the payload:

```bash
compute run train.py::train --gpu cheap --provider vastai --dry-run
```

3. Start training (preflight prints provider, SKU, rate, estimate; confirm or pass `--yes`):

```bash
compute run train.py::train --gpu cheap --provider vastai --timeout 1200 --wait \
  --args '{"epochs":5,"batch_size":32,"image_size":128}'
```

4. After `succeeded`, download weights:

```bash
compute artifacts list <run_id>
compute artifacts get <run_id> <artifact_id> <version> --out ./weights
```

5. Optional — publish the checkpoint to Hugging Face. The token must be stored **and** referenced:

```bash
compute secrets set hf
compute run train.py::train_and_push --gpu cheap --provider vastai --timeout 1200 --wait
```

`train` does not request the secret, so readers can train on the public dataset without an HF token. `train_and_push` lists `secrets=[Secret.from_name("hf")]`; without that declaration the stored secret is never injected.

## Suggested prompt

```
Use SKILL.md in this repo to train the not-hotdog model on compute.cx.
Dataset: theoriclabs/hot-dog-not-hot-dog
Model: randomly initialized CNN, 3 conv blocks (~180k params), CrossEntropyLoss
Train for 5 epochs at 128x128.
First dry-run train.py::train with --gpu cheap --provider vastai.
Then run without --dry-run, show me the preflight GPU and dollar estimate, and ask before confirming spend.
After success, list and download artifacts.
```

## Notes

- Public dataset: no HF token required to train (`train.py::train`).
- Hub publish needs `compute secrets set hf` plus `train.py::train_and_push`.
- To train longer: raise `epochs` and `--timeout` (max 24h). You pay for billed minutes.
- Data lives on Hugging Face, not on compute. Machines are ephemeral.
- No wandb in this guide. No hosted inference API in this guide.

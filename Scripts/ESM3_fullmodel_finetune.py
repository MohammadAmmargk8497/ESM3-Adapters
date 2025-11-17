#!/usr/bin/env python3
import csv
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from esm.data import FastaBatchedDataset
from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def collate_fn(batch, tokenizer, device):
    _, seqs = zip(*batch)
    enc = tokenizer(list(seqs), return_tensors="pt", padding=True)
    return enc["input_ids"].to(device)


def apply_fixed_masking(tokens, positions, mask_token_id):
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    seq_len = tokens.size(1)
    for p in positions:
        if 0 <= p < seq_len:
            mask[:, p] = True
    masked = tokens.clone()
    masked[mask] = mask_token_id
    return masked, mask


def load_loader(fasta_path, tokenizer, batch_size, device):
    ds = FastaBatchedDataset.from_file(str(fasta_path))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, device),
    )


def evaluate(model, loader, mask_positions, mask_token_id, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for tokens in loader:
            masked_tokens, mask = apply_fixed_masking(
                tokens, mask_positions, mask_token_id
            )
            out = model(
                sequence_tokens=masked_tokens,
                average_plddt=torch.ones_like(
                    tokens, dtype=torch.float32, device=device
                ),
                per_res_plddt=torch.zeros_like(
                    tokens, dtype=torch.float32, device=device
                ),
            )
            logits = out.sequence_logits
            masked_logits = logits[mask]
            targets = tokens[mask]
            loss = loss_fn(masked_logits, targets)
            total_loss += loss.item() * targets.numel()
            preds = masked_logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.numel()
    model.train()
    return total_loss / total_tokens, total_correct / total_tokens


@hydra.main(config_path="../conf", config_name="finetune")
def main(cfg: DictConfig):
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    tokenizer = EsmSequenceTokenizer()
    mask_token_id = tokenizer.mask_token_id

    Path(cfg.log_dir).mkdir(exist_ok=True, parents=True)
    Path(cfg.model_save_dir).mkdir(exist_ok=True, parents=True)

    train_loader = load_loader(cfg.train_fasta, tokenizer, cfg.batch_size, device)
    test_loader = load_loader(cfg.test_fasta, tokenizer, cfg.batch_size, device)

    model = (
        ESM3.from_pretrained(cfg.model_name, device=device)
        .to(device)
        .to(torch.float32)
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    with open(cfg.train_log_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "batch_index", "train_loss", "train_accuracy"])

        for epoch in range(cfg.epochs):
            running_loss = 0.0
            running_correct = 0
            running_tokens = 0

            for batch_idx, tokens in enumerate(train_loader):
                masked_tokens, mask = apply_fixed_masking(
                    tokens, cfg.mask_positions, mask_token_id
                )
                out = model(
                    sequence_tokens=masked_tokens,
                    average_plddt=torch.ones_like(
                        tokens, dtype=torch.float32, device=device
                    ),
                    per_res_plddt=torch.zeros_like(
                        tokens, dtype=torch.float32, device=device
                    ),
                )
                logits = out.sequence_logits
                masked_logits = logits[mask]
                targets = tokens[mask]

                loss = loss_fn(masked_logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                preds = masked_logits.argmax(dim=-1)
                correct = (preds == targets).sum().item()
                n_tokens = targets.numel()

                running_loss += loss_val * n_tokens
                running_correct += correct
                running_tokens += n_tokens

                if batch_idx % cfg.eval_epoch_interval == 0:
                    acc = running_correct / running_tokens
                    writer.writerow(
                        [epoch, batch_idx, f"{loss_val:.4f}", f"{acc:.4f}"]
                    )
                    csvfile.flush()
                    logging.info(
                        f"epoch {epoch} batch {batch_idx} loss {loss_val:.4f} acc {acc:.4f}"
                    )
                    print(
                        f"epoch {epoch} batch {batch_idx} train_loss {loss_val:.4f} train_acc {acc:.4f}"
                    )

            if (epoch + 1) % cfg.eval_epoch_interval == 0:
                test_loss, test_acc = evaluate(
                    model, test_loader, cfg.mask_positions, mask_token_id, device
                )
                logging.info(
                    f"epoch {epoch + 1} test_loss {test_loss:.4f} test_acc {test_acc:.4f}"
                )
                ckpt_path = Path(cfg.model_save_dir) / f"{cfg.base_save_name}.pt"
                torch.save(model.state_dict(), str(ckpt_path))
                logging.info(f"saved checkpoint to {ckpt_path}")
                print(
                    f"epoch {epoch + 1} test_loss {test_loss:.4f} test_acc {test_acc:.4f}"
                )


if __name__ == "__main__":
    main()

import csv

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from esm.data import FastaBatchedDataset
from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device) -> ESM3:
    print("Loading Model")
    loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict):
        model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
        model.load_state_dict(loaded)
    else:
        model = loaded
    model.to(device).to(torch.float32)
    model.eval()
    print("Model Load Complete")
    return model


def get_dataloader(fasta_path: str, batch_size: int, device: torch.device):
    tokenizer = EsmSequenceTokenizer()

    def collate_fn(batch):
        labels, seqs = zip(*batch)
        enc = tokenizer(list(seqs), return_tensors="pt", padding=True)
        tokens = enc["input_ids"].to(device)
        return labels, seqs, tokens

    dataset = FastaBatchedDataset.from_file(fasta_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return loader, tokenizer


def apply_fixed_masking(tokens: torch.Tensor, positions: list[int], mask_token_id: int):
    print("Applying Fixed Masking")
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    seq_len = tokens.size(1)
    valid = [p for p in positions if 0 <= p < seq_len]
    for p in valid:
        mask[:, p] = True
    masked = tokens.clone()
    masked[mask] = mask_token_id
    return masked, mask


def apply_random_masking(tokens: torch.Tensor, mask_token_id: int, mask_prob: float):
    print("Applying Random Masking")
    masked = tokens.clone()
    mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    mask[:, 0] = False
    mask[:, -1] = False
    masked[mask] = mask_token_id
    return masked, mask


def evaluate(
    model: ESM3,
    tokenizer: EsmSequenceTokenizer,
    dataloader: DataLoader,
    cfg: DictConfig,
):
    pad_id = tokenizer.pad_token_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_loss_value = 0.0
    total_masked_tokens = 0
    total_correct_preds = 0

    with open(cfg.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "sequence_id",
                "position",
                "original_token",
                "predicted_token",
                "batch_loss",
            ]
        )
        print("Starting Evaluation")
        with torch.no_grad():
            for _, (labels, _, tokens) in enumerate(dataloader):
                if cfg.use_fixed_masking:
                    masked_tokens, mask = apply_fixed_masking(
                        tokens, cfg.mask_positions, tokenizer.mask_token_id
                    )
                else:
                    masked_tokens, mask = apply_random_masking(
                        tokens, tokenizer.mask_token_id, cfg.mask_prob
                    )

                if mask.sum().item() == 0:
                    continue

                out = model(sequence_tokens=masked_tokens)
                logits = out.sequence_logits

                masked_logits = logits[mask]
                targets = tokens[mask]

                loss_val = loss_fn(masked_logits, targets)
                num_masked = targets.size(0)

                total_loss_value += loss_val.item() * num_masked
                total_masked_tokens += num_masked

                pred_indices = masked_logits.argmax(dim=-1)
                total_correct_preds += (pred_indices == targets).sum().item()

                orig_tokens = tokenizer.convert_ids_to_tokens(targets.tolist())
                pred_tokens = tokenizer.convert_ids_to_tokens(pred_indices.tolist())

                positions = mask.nonzero(as_tuple=False)
                for idx in range(positions.size(0)):
                    b, pos = positions[idx].tolist()
                    seq_id = labels[b]
                    writer.writerow(
                        [
                            seq_id,
                            pos,
                            orig_tokens[idx],
                            pred_tokens[idx],
                            f"{loss_val.item():.4f}",
                        ]
                    )

        if total_masked_tokens > 0:
            avg_loss = total_loss_value / total_masked_tokens
            accuracy = total_correct_preds / total_masked_tokens
            print(f"Test Loss {avg_loss:.4f}, Test Accuracy {accuracy:.4f}")
        else:
            print("No masked tokens were evaluated; cannot compute loss or accuracy.")


@hydra.main(config_path="../conf", config_name="inference")
def main(cfg: DictConfig):
    device = get_device()
    model = load_model(cfg.checkpoint_path, device)
    dataloader, tokenizer = get_dataloader(cfg.test_fasta_path, cfg.batch_size, device)
    evaluate(model, tokenizer, dataloader, cfg)
    print("Completed Evaluation")


if __name__ == "__main__":
    main()

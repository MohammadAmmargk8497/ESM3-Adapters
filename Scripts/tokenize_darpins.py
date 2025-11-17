import torch
from torch.utils.data import DataLoader

from esm.data import FastaBatchedDataset
from esm.pretrained import load_local_model
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

model_name = "esm3_sm_open_v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_local_model(model_name, device)
model.eval()

# tokenizer
tokenizer = EsmSequenceTokenizer()


# raw sequences -> padded token tensors
def batch_converter(batch):
    labels, seqs = zip(*batch)
    encoding = tokenizer(list(seqs), return_tensors="pt", padding=True)
    tokens = encoding["input_ids"].to(device)
    return labels, seqs, tokens


# FASTA data
fasta_path = "esm3-darpins/fasta/darpin.fasta"
dataset = FastaBatchedDataset.from_file(fasta_path)
data_loader = DataLoader(dataset, batch_size=8, collate_fn=batch_converter)

for batch_idx, (labels, seq_strs, tokens) in enumerate(data_loader):
    print(f"Batch {batch_idx}")
    print("Labels:", labels)
    print("Original sequences:", seq_strs)
    print("Tokens shape:", tokens.shape)  # (batch_size, max_seq_length)
    break

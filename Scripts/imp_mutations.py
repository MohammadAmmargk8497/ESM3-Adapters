#!/usr/bin/env python3
from pathlib import Path


def parse_fasta(path):
    headers, sequences = [], []
    with open(path, "r") as f:
        header, seq = None, ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    headers.append(header)
                    sequences.append(seq)
                header = line[1:]
                seq = ""
            else:
                seq += line
        if header is not None:
            headers.append(header)
            sequences.append(seq)
    return headers, sequences


def compute_mutation_counts(seqs):
    seq_length = len(seqs[0])
    counts = [0] * seq_length

    for ref_seq in seqs:
        for seq in seqs:
            if seq is ref_seq:
                continue
            for i, (a, b) in enumerate(zip(ref_seq, seq)):
                if a != b:
                    counts[i] += 1

    return counts


def top_n_positions(counts, n=50):
    pos_counts = [(i, c) for i, c in enumerate(counts)]
    pos_counts.sort(key=lambda x: x[1], reverse=True)
    return [pos for pos, _ in pos_counts[:n]]


if __name__ == "__main__":
    fasta_path = Path("/content/filtered_sequences.fasta")
    headers, seqs = parse_fasta(fasta_path)
    counts = compute_mutation_counts(seqs)
    MASK_POSITIONS = top_n_positions(counts, n=50)
    print("Top mutation positions :")
    print(MASK_POSITIONS)

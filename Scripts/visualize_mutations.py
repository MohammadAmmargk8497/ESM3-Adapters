import math
import os

import matplotlib.pyplot as plt
import pandas as pd

# Constants
FASTA_FILENAME = "/content/filtered_sequences.fasta"
OUTPUT_DIR = "/scratch/user/zahidhussain909/esm3-darpins/Scripts/logs"


# Parse FASTA into headers and sequences
def parse_fasta(path):
    headers = []
    sequences = []
    with open(path) as f:
        header = None
        seq = ""
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


# Plotting mutation
def plot_mutation_subplots(seqs, headers, start_idx, end_idx, output_path=None):
    if start_idx < 0 or end_idx >= len(seqs) or start_idx > end_idx:
        raise IndexError(
            "Invalid index range. Must satisfy 0 <= start_idx <= end_idx < number of sequences."
        )

    num_refs = end_idx - start_idx + 1
    seq_length = len(seqs[0])

    ncols = math.ceil(math.sqrt(num_refs))
    nrows = math.ceil(num_refs / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for idx_offset, ref_idx in enumerate(range(start_idx, end_idx + 1)):
        reference = seqs[ref_idx]
        mutation_counts = [0] * seq_length

        for seq_idx, seq in enumerate(seqs):
            if seq_idx == ref_idx:
                continue
            for i in range(seq_length):
                if seq[i] != reference[i]:
                    mutation_counts[i] += 1

        ax = axes[idx_offset]
        ax.bar(range(1, seq_length + 1), mutation_counts)
        ax.set_title(f"Ref #{ref_idx}: {headers[ref_idx]}", fontsize=8)

        if idx_offset % ncols == 0:
            ax.set_ylabel("Mut Count")
        if idx_offset // ncols == nrows - 1:
            ax.set_xlabel("Position")
        ax.set_xlim(1, seq_length)

    for extra_idx in range(num_refs, len(axes)):
        fig.delaxes(axes[extra_idx])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close(fig)


# Extract top mutations
def get_top_mutation_dataframe(
    seqs, headers, start_idx, end_idx, top_n=10, output_path=None
):
    if start_idx < 0 or end_idx >= len(seqs) or start_idx > end_idx:
        raise IndexError(
            "Invalid index range. Must satisfy 0 <= start_idx <= end_idx < number of sequences."
        )

    seq_length = len(seqs[0])
    records = []

    for ref_idx in range(start_idx, end_idx + 1):
        reference = seqs[ref_idx]
        mutation_counts = [0] * seq_length

        for seq_idx, seq in enumerate(seqs):
            if seq_idx == ref_idx:
                continue
            for i in range(seq_length):
                if seq[i] != reference[i]:
                    mutation_counts[i] += 1

        sorted_positions = sorted(
            [(i + 1, count) for i, count in enumerate(mutation_counts)],
            key=lambda x: x[1],
            reverse=True,
        )

        for pos, count in sorted_positions[:top_n]:
            records.append(
                {
                    "Ref Index": ref_idx,
                    "Ref Header": headers[ref_idx],
                    "Position": pos,
                    "Mutation Count": count,
                    "Ref Token": reference[pos - 1],
                }
            )

    df = pd.DataFrame(records)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    # === CONFIG ===
    do_plot = True
    do_print = True
    plot_start_idx = 0
    plot_end_idx = 25
    print_start_idx = 0
    print_end_idx = 8
    top_n_mutations = 10
    # ==============

    if not os.path.exists(FASTA_FILENAME):
        raise FileNotFoundError(f"Upload '{FASTA_FILENAME}' and rerun.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    headers, seqs = parse_fasta(FASTA_FILENAME)

    if do_plot:
        plot_file = os.path.join(OUTPUT_DIR, "mutation_plot.png")
        plot_mutation_subplots(
            seqs, headers, plot_start_idx, plot_end_idx, output_path=plot_file
        )

    if do_print:
        csv_file = os.path.join(OUTPUT_DIR, "top_mutations.csv")
        df_top_mutations = get_top_mutation_dataframe(
            seqs,
            headers,
            print_start_idx,
            print_end_idx,
            top_n=top_n_mutations,
            output_path=csv_file,
        )
        print(df_top_mutations.tail(50))

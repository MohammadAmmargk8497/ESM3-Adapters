import pandas as pd

df = pd.read_csv("esm3-darpins/ZC010423cleaned.csv")
df_sorted = df.sort_values(by="Score", ascending=False)
top_100 = df_sorted.head(100)

with open("esm3-darpins/fasta/darpin.fasta", "w") as f:
    for idx, seq_ in enumerate(top_100["Sequence"].tolist(), start=1):
        file_id = f"dar_{idx:03d}"
        f.write(f">{file_id}\n")
        f.write(seq_ + "\n")

import torch

from esm.models.esm3 import ESM3

client = ESM3.from_pretrained("esm3_sm_open_v1")  # downloads weights + config
alphabet = client.alphabet
torch.save(
    {"model": client, "alphabet": alphabet},
    "/scratch/user/zahidhussain909/Dr_Tao/ESM/models/esm3_sm_open_v1_model2.pth",
)

import torch
import numpy as np
from dataset import Sun360PreExtractDataset
import tqdm
import os
from utils.util import to_code

device = "cuda"

vqvaes = torch.load("vqvae_models.pth")
vqvaes = [i.to(device) for i in vqvaes]
encoder,decoder,quantizer = [i.eval() for i in vqvaes]
img_path = "../sun360_outdoor" if os.path.exists("../sun360_outdoor") else "../../sun360_outdoor"
os.makedirs("datas_60_outdoor",exist_ok=True)
os.makedirs("datas_test_60_outdoor",exist_ok=True)


with torch.autocast(device_type='cuda', dtype=torch.float16):
    dataset = Sun360PreExtractDataset(img_path, train=True,extract=True)
    for i,d in enumerate(tqdm.tqdm(dataset)):
        with torch.no_grad():
            data,path,deg = d
            code = to_code(data,encoder,quantizer,device)
            code = code.reshape(-1, 16 * 16 * 26)
            np.savez_compressed(f"datas_60_outdoor/{i:06}.npz",data=code,path=np.array([path]),degree=np.array([deg]))

    dataset = Sun360PreExtractDataset(img_path, train=False, extract=True)
    for i,d in enumerate(tqdm.tqdm(dataset)):
        with torch.no_grad():
            data,path,deg = d
            code = to_code(data,encoder,quantizer,device)
            code = code.reshape(-1, 16 * 16 * 26)
            np.savez_compressed(f"datas_test_60_outdoor/{i:06}.npz",data=code,path=np.array([path]),degree=np.array([deg]))

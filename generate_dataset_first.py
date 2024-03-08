import torch
import numpy as np
from dataset import Sun360PreExtractDataset
from torchvision.transforms.functional import to_pil_image
import tqdm
import os
from utils.util import to_code,code_to_img
device = "cpu"

vqvaes = torch.load("vqvae_models.pth")
vqvaes = [i.to(device) for i in vqvaes]
encoder,decoder,quantizer = [i.eval() for i in vqvaes]
img_path = "../sun360_outdoor" if os.path.exists("../sun360_outdoor") else "../../sun360_outdoor"
dataset = Sun360PreExtractDataset(img_path, train=True,extract=False)
dataset_test = Sun360PreExtractDataset(img_path, train=False,extract=False)
os.makedirs("datas_all_outdoor",exist_ok=True)
os.makedirs("datas_all_test_outdoor",exist_ok=True)
os.makedirs("vqvae_recon_outdoor",exist_ok=True)
os.makedirs("vqvae_recon_outdoor_test",exist_ok=True)

torch.backends.cudnn.benchmark=True

with torch.inference_mode():
    for i,d in enumerate(tqdm.tqdm(dataset)):
        data,path,roll_pixel = d
        data = to_code(data,encoder,quantizer,device)
        np.savez_compressed(f"datas_all_outdoor/{i:06}.npz",data=data,path=np.array([path]),degree=np.array([roll_pixel]))
        if roll_pixel==0:
            img = to_pil_image(code_to_img(data,quantizer,decoder,device)[0])
            img.save(f"vqvae_recon_outdoor/{os.path.basename(path)}")

with torch.inference_mode():
    for i,d in enumerate(tqdm.tqdm(dataset_test)):
        data,path,roll_pixel = d
        data = to_code(data,encoder,quantizer,device)
        np.savez_compressed(f"datas_all_test_outdoor/{i:06}.npz",data=data,path=np.array([path]),degree=np.array([roll_pixel]))
        if roll_pixel==0:
            img = to_pil_image(code_to_img(data,quantizer,decoder,device)[0])
            img.save(f"vqvae_recon_outdoor_test/{os.path.basename(path)}")

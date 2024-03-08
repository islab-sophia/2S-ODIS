import math

import numpy as np
import torch
import torch.optim as optim
import argparse
import torchvision
from utils.image_extract import ExtractImageConvertor
from models.second_stage_module import SecondStageModel
from dataset import SecondStageDataset
from torch.utils.data import DataLoader
from maskgit_sampler import MaskGitSampler
import mlflow
import os
import shutil
import tqdm
torch.backends.cudnn.benchmark=True


tmp_path = "./tmp/second_stage"
if os.path.exists(tmp_path):
    shutil.rmtree(tmp_path)
os.makedirs(tmp_path, exist_ok=True)

mlflow_path = "./mlruns/"
mlflow.set_tracking_uri(mlflow_path)
mlflow.set_experiment('SecondStage')

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--n_updates", type=int, default=180000)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--log_interval", type=int, default=5000)
parser.add_argument("--model_dim", type=int, default=256)
parser.add_argument("--layer_nums", nargs="*", type=int, default=[8])
parser.add_argument("--data_path", type=str, default="datas_60_outdoor")
parser.add_argument("--data_test_path", type=str, default="datas_test_60_outdoor")
parser.add_argument("--vqrecon_img_path", type=str, default="vqvae_recon_outdoor" )
parser.add_argument("--vqrecon_img_test_path", type=str, default="vqvae_recon_outdoor_test" )
parser.add_argument("--layer_const", nargs="*", type=str, default=["maxvit"])
parser.add_argument("--disable_amp", action='store_true')
parser.add_argument("--bf16", action='store_true')
args = parser.parse_args()

mlflow.log_params(vars(args))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
sampler = MaskGitSampler(1024,is_second_stage=True).to("cuda")


"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

_,decoder,quantizer = torch.load("vqvae_models.pth")
decoder.eval(),quantizer.eval()
gen = SecondStageModel(args.model_dim,
           layer_const=args.layer_const,
           block_nums=args.layer_nums,view_angle=60).to(device)
gen.train()

scaler = torch.cuda.amp.GradScaler()

quantizer.cpu()
decoder.cpu()
real_img_path = "../sun360_outdoor" if os.path.exists("../sun360_outdoor") else "../../sun360_outdoor"
for i in ["./","../","../../",]:
    dataset = SecondStageDataset(i+args.data_path,i+args.vqrecon_img_path,real_img_path, train=True)
    test_dataset = SecondStageDataset(i+args.data_test_path,i+args.vqrecon_img_test_path,real_img_path, train=False)
    if len(dataset)>0:
        break
dataloader = DataLoader(dataset=dataset, num_workers=2, batch_size=args.batch_size,
                                      shuffle=True, drop_last=True,)
dataloader_test = DataLoader(dataset=test_dataset, num_workers=2, batch_size=12,
                                      shuffle=True, drop_last=True,)

"""
Set up optimizer and training loop
"""
gen_optimizer = optim.AdamW(gen.parameters(), lr=args.learning_rate,weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=0.95)

eic = ExtractImageConvertor(view_angle=60)

log_metrics = ["loss"]
reg_loss = torch.nn.CrossEntropyLoss()
gen_optimizer.zero_grad()
_,generate_data_cpu,mask_image_cpu = next(iter(dataloader_test))
generate_data_cpu = generate_data_cpu.clone()
mask_image_cpu = mask_image_cpu.clone()
imgs = torchvision.utils.make_grid(torch.clip(generate_data_cpu,0,1),nrow=4)
mask_imgs = torchvision.utils.make_grid(torch.clip(mask_image_cpu[:,:3],0,1),nrow=4)
mlflow.log_image(imgs.permute(1, 2, 0).cpu().numpy(), f"test_cond_image_preview.jpg")
mlflow.log_image(mask_imgs.permute(1, 2, 0).cpu().numpy(), f"test_mask_image_preview.jpg")

def train():
    losses = {}
    for i in log_metrics:
        losses[i] = []

    for i in tqdm.tqdm(range(args.n_updates)):
        with torch.amp.autocast('cuda', dtype=amp_dtype,enabled=not args.disable_amp):
            x,cond,mask_image = next(iter(dataloader))
            x = x.reshape(-1,16,16)
            x = x.to(device)
            cond = cond.to(device)
            mask_image = mask_image.to(device)

            x_noise = sampler.noise(x.clone())
            x_recon = gen(x_noise,cond,mask_image)
            loss = reg_loss(x_recon, x)
        scaler.scale(loss).backward()
        scaler.step(gen_optimizer)
        # スケールの更新
        scaler.update()
        losses["loss"].append(loss.item())
        del loss,x_recon,x_noise,x,cond

        if i % args.log_interval == 0 or i==args.n_updates-1:
            """
            save model and print values
            """
            quantizer.to(device)
            decoder.to(device)
            generate_data = generate_data_cpu.to(device)
            mask_image = mask_image_cpu.to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16,enabled=not args.disable_amp):
                with torch.no_grad():
                    x = sampler.sampling(gen,generate_data.shape[0],device,[generate_data,mask_image])
                    z_q = quantizer.get_codebook_entry(x, (-1, 16, 16, 256))
                    z_q = z_q.reshape(-1, 256, 16, 16)
                    recons = []
                    for j in range(z_q.shape[0]):
                        recon = decoder(z_q[[j]])
                        recons.append(recon)
                    recons = torch.cat(recons,dim=0)
                    recons = torch.clip(recons / 2 + 0.5, 0, 1)
                    imgs = eic(recons.cpu().to(torch.float32)).permute(0,2,3,1).numpy()
                    imgs = torch.from_numpy(imgs).permute(0,3,1,2)
                    imgs = torchvision.utils.make_grid(torch.clip(imgs,0,1),nrow=4)
                    mlflow.log_image(imgs.permute(1,2,0).cpu().numpy(), f"epoch_{i:0>6}_test_preview.jpg")
                    del imgs,z_q,recon,recons,x,generate_data,mask_image
            decoder.cpu()
            quantizer.cpu()
            model_out_path = os.path.join(tmp_path, f"SecondStage_epoch_{i:0>6}.pth")
            torch.save(gen, model_out_path)
            mlflow.log_artifacts(tmp_path)

            for j in log_metrics:
                mlflow.log_metrics({j:np.mean(losses[j])}, step=i)

            losses = {}
            for j in log_metrics:
                losses[j] = []
            scheduler.step()

if __name__ == "__main__":
    train()

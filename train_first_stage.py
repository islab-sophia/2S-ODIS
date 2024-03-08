import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import torchvision
from models.first_stage_module import FirstStageModel
from dataset import FirstStageDataset
from maskgit_sampler import MaskGitSampler
import mlflow
import os
import shutil
import tqdm
torch.backends.cudnn.benchmark=True

mlflow_path = "./mlruns/"
mlflow.set_tracking_uri(mlflow_path)
tmp_path = "./tmp/first_stage"
if os.path.exists(tmp_path):
    shutil.rmtree(tmp_path)
os.makedirs(tmp_path, exist_ok=True)
mlflow.set_experiment('FirstStage')

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--n_updates", type=int, default=180000)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--log_interval", type=int, default=10000)
parser.add_argument("--model_dim", type=int, default=128)
parser.add_argument("--layer_num", type=int, default=8)
parser.add_argument("--deepconv", action='store_true')

args = parser.parse_args()

mlflow.log_params(vars(args))
gamma = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"
sampler = MaskGitSampler(1024,is_second_stage=False).to("cuda")

"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

_,decoder,quantizer = torch.load("vqvae_models.pth")
decoder.eval(),quantizer.eval()

gen = FirstStageModel(1024,args.model_dim,args.layer_num,
                                seq_len=512,deep_conv=args.deepconv).to(device)
gen.train()

quantizer.cpu()
decoder.cpu()
img_path = "../sun360_outdoor" if os.path.exists("../sun360_outdoor") else "../../sun360_outdoor"
for i in ["./","../","../../",]:
    dataset = FirstStageDataset(i+"datas_all_outdoor",img_path, train=True)
    test_dataset = FirstStageDataset(i+"datas_all_test_outdoor",img_path, train=False)
    if len(dataset)>0:
        break
dataloader = DataLoader(dataset=dataset, num_workers=2, batch_size=args.batch_size,
                                      shuffle=True, drop_last=True,)
dataloader_test = DataLoader(dataset=test_dataset, num_workers=2, batch_size=16,
                                      shuffle=False, drop_last=False,)

"""
Set up optimizer and training loop
"""
gen_optimizer = optim.AdamW(gen.parameters(), lr=args.learning_rate,weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=0.95)

log_metrics = ["loss"]
reg_loss = torch.nn.CrossEntropyLoss()
gen_optimizer.zero_grad()
_,generate_data = next(iter(dataloader_test))
generate_data = generate_data.to("cpu")
# a = a.to(device)
def train():
    losses = {}
    for i in log_metrics:
        losses[i] = []

    for i in tqdm.tqdm(range(args.n_updates)):

        x,cond = next(iter(dataloader))
        x = x.to(device)
        cond = cond.to(device)
        x_noise = sampler.noise(x.clone())
        x_recon = gen(x_noise,cond).permute(0,2,1)
        loss = reg_loss(x_recon, x)
        gen_optimizer.zero_grad()
        loss.backward()
        gen_optimizer.step()

        losses["loss"].append(loss.item())

        if i % args.log_interval == 0 or i==args.n_updates-1:
            """
            save model and print values
            """
            quantizer.to(device)
            decoder.to(device)
            with torch.no_grad():
                x = sampler.sampling(gen,generate_data.shape[0],device,[generate_data.clone().to(device)])
                z_q = quantizer.get_codebook_entry(x, (-1, 16, 32, 256))
                z_q = z_q.reshape(-1, 256, 16, 32)
                recons = decoder(z_q)
                imgs = torchvision.utils.make_grid(torch.clip(recons/2+0.5,0,1),nrow=4)
                mlflow.log_image(imgs.permute(1,2,0).cpu().numpy(), f"epoch_{i:0>6}_test_preview.jpg")
                del imgs,x
            decoder.cpu()
            quantizer.cpu()
            model_out_path = os.path.join(tmp_path, f"FirstStage_epoch_{i:0>6}.pth")
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

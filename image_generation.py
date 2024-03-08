import glob
import os.path
import numpy as np
import torch
from tqdm import tqdm
import argparse
from utils.image_extract import ExtractImageConvertor
from PIL import Image
from maskgit_sampler import MaskGitSampler
torch.backends.cudnn.benchmark=True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sampler_first = MaskGitSampler(1024).to(device)
sampler_second = MaskGitSampler(1024,is_second_stage=True).to(device)

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="./generated_img")
parser.add_argument("--lowres_base_path", type=str, default="./")
parser.add_argument("--lowres_name", type=str,default="FirstStage_epoch_179999.pth")
parser.add_argument("--highres_base_path", type=str, default="./")
parser.add_argument("--highres_name", type=str,default="SecondStage_epoch_179999.pth")
parser.add_argument("--experiment_name", type=str,default="2S-ODIS_test")
parser.add_argument("--mask_path", type=str,default="mask_img.png")
args = parser.parse_args()

print(vars(args))

save_path = args.save_path
lowres_model_path = os.path.join(args.lowres_base_path,args.lowres_name)
highres_model_path = os.path.join(args.highres_base_path,args.highres_name)
experiment_name = args.experiment_name

encoder,decoder,quantizer = torch.load("vqvae_models.pth")
decoder.eval(),quantizer.eval()

lowres_model = torch.load(lowres_model_path).to(device)
highres_model = torch.load(highres_model_path).to(device)

quantizer.to(device)
decoder.to(device)
imgs = glob.glob("../sun360_outdoor/test/*.jpg")


def decode_images(x, z_shape=(16, 16), batch_size=4):
    quantizer.to(device).eval()
    decoder.to(device).eval()
    z_q = quantizer.get_codebook_entry(x, (-1, z_shape[0], z_shape[1], 256))
    z_q = z_q.reshape(-1, 256, z_shape[0], z_shape[1])
    ls = []
    for i in range((x.shape[0] - 1) // batch_size + 1):
        recons = decoder(z_q[i * batch_size:min((i + 1) * batch_size, x.shape[0])])
        ls.append(recons)
    recons = torch.concat(ls, dim=0)
    quantizer.cpu()
    decoder.cpu()
    return recons

mask_img = torch.from_numpy(np.array(Image.open(args.mask_path))/255.0).unsqueeze(0)
tensor_init = lambda x: x.to(torch.float32).unsqueeze(0).to(device)
batch_size = 1

os.makedirs(os.path.join(save_path,experiment_name+"_lowres"),exist_ok=True)
os.makedirs(os.path.join(save_path,experiment_name),exist_ok=True)
eic = ExtractImageConvertor(erp_image_shape=(1024,512),view_angle=60)

def process_images(img_path):
    target_image = torch.from_numpy(np.array(Image.open(img_path).resize((512, 256))) / 255.0).permute(2, 0, 1)
    embbed_image = mask_img * target_image
    basename = os.path.basename(img_path)
    condition_image = torch.cat([embbed_image, mask_img], dim=0)

    with torch.no_grad():
        condition_image = tensor_init(condition_image).repeat(batch_size, 1, 1, 1)
        x = sampler_first.sampling(lowres_model, batch_size, device, [condition_image])
        recons = decode_images(x, z_shape=(16, 32))
        recons = torch.clip(recons / 2 + 0.5, 0, 1)
        Image.fromarray(torch.clip(recons[0] * 255, 0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save(
            os.path.join(save_path, experiment_name + "_lowres", basename))
        x = sampler_second.sampling(highres_model, batch_size,device,[recons, condition_image])
        recons = decode_images(x, batch_size=1).cpu().to(torch.float32)
        recons = torch.clip(recons / 2 + 0.5, 0, 1)
        imgs = eic(recons)
        Image.fromarray(torch.clip(imgs[0] * 255, 0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save(
            os.path.join(save_path, experiment_name, basename))


for i in tqdm(imgs):
    process_images(i)
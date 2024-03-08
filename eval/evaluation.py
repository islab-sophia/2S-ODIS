import torch
import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from cleanfid import fid
import lpips
import json
from torchmetrics.image.inception import InceptionScore

_ = torch.manual_seed(123)


def calc_is(img_path,batch=16):
    inception = InceptionScore()
    img_ls = glob.glob(os.path.join(img_path,"*.jpg")) + glob.glob(os.path.join(img_path,"*.png"))
    loops = (len(img_ls)-1)//batch+1

    for i in tqdm(range(loops)):
        input_img_path = img_ls[batch*i:min(batch*(i+1),len(img_ls))]
        input_imgs = torch.stack([torch.from_numpy(np.array(Image.open(i))).permute(2, 0, 1) for i in input_img_path])
        inception.update(input_imgs)
    return [i.item() for i in inception.compute()]

def calc_fid(img_path,base_dataset = "sun360_outdoor"):
    # img_path = f"../{base_dataset}" if os.path.exists(f"../{base_dataset}") else f"../../{base_dataset}"
    # test_path = os.path.join(img_path, "test")
    # fid.make_custom_stats(f"{base_dataset}_test", test_path)
    score = fid.compute_fid(img_path, dataset_name=f"{base_dataset}_test", dataset_split="custom")
    return score.item()

@torch.no_grad()
def calc_lpips(img_path,all_pairs=False):
    loss_fn = lpips.LPIPS(net='alex', version="0.1")
    loss_fn.cuda()
    files = glob.glob(os.path.join(img_path,"*.jpg")) + glob.glob(os.path.join(img_path,"*.png"))
    dists = []
    for (ff, file) in enumerate(tqdm(files[:-1])):
        img0 = lpips.im2tensor(lpips.load_image(file))  # RGB image from [-1,1]
        img0 = img0.cuda()

        if all_pairs:
            files1 = files[ff + 1:]
        else:
            files1 = [files[ff + 1], ]

        for file1 in files1:
            img1 = lpips.im2tensor(lpips.load_image(file1))
            img1 = img1.cuda()

            if img0.shape != img1.shape:
                continue

            # Compute distance
            dist01 = loss_fn.forward(img0, img1)
            dists.append(dist01.item())

    avg_dist = np.mean(np.array(dists))
    stderr_dist = np.std(np.array(dists)) / np.sqrt(len(dists))
    return avg_dist,stderr_dist

def calc_metrics(path="../../sun360_outdoor/test"):
    is_avg,is_std = calc_is(path)
    fid = calc_fid(path)
    lpips_avg,lpips_std = calc_lpips(path)
    return {
        "IS_avg":is_avg,
        "IS_std":is_std,
        "FID":fid,
        "LPIPS_avg":lpips_avg,
        "LPIPS_std":lpips_std,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--image_path", type=str,default="../generated_img/2S-ODIS_test")
    args = parser.parse_args()

    os.makedirs(args.save_path,exist_ok=True)
    metrics = calc_metrics(args.image_path)
    json.dump(metrics,open(os.path.join(args.save_path,os.path.splitext(os.path.basename(args.image_path))[0])+".json","w"))
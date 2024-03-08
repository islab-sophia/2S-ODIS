from cleanfid import fid
import os

base_dataset = "sun360_outdoor"
img_path = f"../{base_dataset}" if os.path.exists(f"../{base_dataset}") else f"../../{base_dataset}"
test_path = os.path.join(img_path,"test")
fid.make_custom_stats(f"{base_dataset}_test",test_path)
score = fid.compute_fid(test_path,dataset_name=f"{base_dataset}_test",dataset_split="custom")
assert score < 1
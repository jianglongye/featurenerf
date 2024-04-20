import argparse
import glob
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from dino.dino_model import DINO

from util import set_random_seed

SPLIT_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "splits"))

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default=os.path.join(os.path.dirname(__file__), "..", "data"))
parser.add_argument("--skip_exist", action="store_true")
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument(
    "--synsets", nargs="+", default=["03001627", "04379243", "02691156", "03790512", "02958343", "02876657"]
)
parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
parser.add_argument("--target_views", type=str, default="-1")
args = parser.parse_args()

data_root = args.data_root
synsets = args.synsets
skip_exist = args.skip_exist
img_size = args.img_size
splits = args.splits
target_views = [int(x) for x in args.target_views.split(" ")]
use_all_views = -1 in target_views
assert all([split in ["train", "val", "test"] for split in splits])
print(f"Using synsets: {synsets}")
print(f"Using splits: {splits}")
print(f"Using target views: {target_views}")

assert os.path.exists(data_root), f"Data root {data_root} does not exist!"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

net = DINO().eval()
net = net.to(device)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

set_random_seed(0)
if __name__ == "__main__":
    for synset in synsets:
        print(f"Processing synset {synset}...")
        with open(os.path.join(SPLIT_FILES_DIR, f"{synset}.json")) as f:
            split_file = json.load(f)
        img_dir = os.path.join(data_root, "img", synset)
        assert os.path.exists(img_dir), f"Image directory {img_dir} does not exist!"
        feat_dir = os.path.join(data_root, f"dino_feat_{img_size}", synset)
        os.makedirs(feat_dir, exist_ok=True)

        for split in splits:
            print(f"Processing {split} split...")

            obj_ids = split_file[split]
            for obj_id in tqdm.tqdm(obj_ids):
                feat_path = os.path.join(feat_dir, f"{obj_id}.npz")

                obj_dir = os.path.join(img_dir, obj_id)
                img_paths = sorted(glob.glob(os.path.join(obj_dir, "*.png")))
                if not use_all_views:
                    img_paths = [img_paths[i] for i in target_views]
                if skip_exist and os.path.exists(feat_path):
                    continue

                # https://github.com/applied-ai-lab/zero-shot-pose/blob/main/zsp/method/vision_transformer_flexible.py#L333
                img_tf_op = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
                imgs = [Image.open(img_path).convert("RGB") for img_path in img_paths]
                imgs = torch.stack([img_tf_op(img) for img in imgs]).to(device)

                masks = np.stack([np.asarray(Image.open(img_path))[..., -1:] for img_path in img_paths], axis=0)
                masks = torch.from_numpy(masks).to(device).permute(0, 3, 1, 2)

                if imgs.shape[2] != img_size or imgs.shape[3] != img_size:
                    imgs = F.interpolate(imgs, size=(img_size, img_size), mode="bilinear")

                with torch.no_grad():
                    feats = net(imgs)

                down_masks = F.interpolate(masks, size=(feats.shape[2], feats.shape[3]), mode="nearest")
                feats = torch.permute(feats, (0, 2, 3, 1))
                feats[torch.logical_not(down_masks.squeeze(1).bool())] = 0
                feats = torch.permute(feats, (0, 3, 1, 2))

                if use_all_views:
                    np.savez_compressed(feat_path, feats=feats.cpu().numpy())
                else:
                    feats = feats.cpu().numpy()
                    feats = {target_views[i]: feats[i] for i in range(len(target_views))}
                    np.savez_compressed(feat_path, feats=feats)

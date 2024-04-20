import copy
import json
import os
import shutil
import sys

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import cv2
import numpy as np
import seaborn as sns
import skimage
import torch
import torch.nn.functional as F
import tqdm
from pyhocon import ConfigFactory

torch.multiprocessing.set_sharing_strategy("file_system")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import util
from data import get_split_dataset
from model import make_model
from render import NeRFEmbedRenderer


def extra_args(parser):
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_vis", action="store_true")

    parser.add_argument("--output", "-O", type=str, default="outputs")
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

DEVICE = util.get_cuda(args.gpu_id[0])

use_last_feat = True

conf["data"]["format"] = args.dataset_format
output_dir = os.path.join(args.output.strip(), f"eval_feature_nv_all_{args.synset}")
output_dir = os.path.join(output_dir)

assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
split_path = os.path.join(assets_dir, "splits", f"{args.synset}.json")
with open(split_path, "r") as f:
    split_file = json.load(f)


if os.path.exists(output_dir) and input(f"output dir {output_dir} exists, delete? (y/n)") == "y":
    shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

if "dino" in args.conf:
    short_name = "dino_nerf"
elif "diff" in args.conf:
    short_name = "diff_nerf"
else:
    short_name = args.name
model_info = {short_name: {"conf_path": args.conf, "name": args.name}}

nets = {}
for method_name in model_info:
    temp_conf = ConfigFactory.parse_file(model_info[method_name]["conf_path"])
    temp_net = make_model(temp_conf["model"]).to(device=DEVICE)
    temp_args = copy.deepcopy(args)
    temp_args.name = model_info[method_name]["name"]
    temp_net.load_weights(temp_args)
    nets[method_name] = temp_net


all_uni_ids = sorted(list(set([x["model_id"] for x in split_file["cross_view_pairs"]])))
extra_kwargs = {"model_ids": all_uni_ids}
assert args.dataset_format.startswith("feature"), "why not feature dataset?"
extra_kwargs["synset"] = args.synset
dset = get_split_dataset(args.dataset_format, args.datadir, want_split=args.split, training=False, **extra_kwargs)

util.set_random_seed(args.seed)

renderer = NeRFEmbedRenderer.from_conf(conf["renderer"], eval_batch_size=args.ray_batch_size, ret_last_feat=True)
renderer = renderer.to(device=DEVICE)
render_pars = {}
for method_name in nets:
    render_pars[method_name] = renderer.bind_parallel(nets[method_name], args.gpu_id, simple_output=True).eval()


z_near = dset.z_near
z_far = dset.z_far


@torch.no_grad()
def extract_feature_map(img, src_pose, trg_pose, focal, net=None, render_par=None, H=128, W=128, eval_ray_num=10000):
    grid_rays = util.gen_rays(trg_pose.reshape(-1, 4, 4), W, H, focal, z_near, z_far)
    all_rays = grid_rays.reshape(1, -1, 8)

    net.encode(img.to(device=DEVICE), src_pose.to(device=DEVICE), focal.to(device=DEVICE))

    rgb_fine_list, embed_list = [], []
    begin = 0
    while begin < all_rays.shape[1]:
        end = min(begin + eval_ray_num, all_rays.shape[1])
        rgb_fine, _, embed = render_par(all_rays[:, begin:end, :].to(device=DEVICE))
        rgb_fine_list.append(rgb_fine)
        embed_list.append(embed)
        begin = end
    rgb_fine = torch.cat(rgb_fine_list, dim=1)
    feat = torch.cat(embed_list, dim=1)

    norm_feat = feat / torch.maximum(feat.norm(dim=-1, keepdim=True), torch.tensor(1e-6).to(feat.device))
    norm_feat = norm_feat.reshape(-1, norm_feat.shape[-1])

    return norm_feat, rgb_fine


def seg_map_to_vis_map(seg_map, text=None):
    def _hex_to_rgb(hex):
        return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))

    palette = ["70d6ff", "ff70a6", "ff9770", "ffd670", "e9ff70"]
    palette = [_hex_to_rgb(x) for x in palette]

    unique_seg_labels = np.unique(seg_map)

    label_to_color = np.zeros((max(unique_seg_labels) + 1, 3), dtype=np.uint8)
    palette_idx = 0
    for seg_label in unique_seg_labels:
        if seg_label == 0:
            label_to_color[seg_label] = (255, 255, 255)
        else:
            label_to_color[seg_label] = palette[palette_idx % len(palette)]
            palette_idx += 1

    vis_map = label_to_color[seg_map]

    if text is not None:
        vis_map = cv2.putText(vis_map.copy(), text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis_map


def draw_kps(img, kps, kp_labels, text=None):
    colorpalette = sns.color_palette(n_colors=len(kp_labels))

    vis_img = img.copy()
    for kp_idx in range(kps.shape[0]):
        x, y, kp_id = kps[kp_idx].tolist()
        if kp_id not in kp_labels:
            continue
        color = colorpalette[kp_labels.index(kp_id)]
        color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        # boundary_color = (color[0] // 2, color[1] // 2, color[2] // 2)
        # vis_img = cv2.circle(vis_img, (int(x), int(y)), 3, boundary_color, 1, cv2.LINE_AA)
        vis_img = cv2.circle(vis_img, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)

    if text is not None:
        vis_img = cv2.putText(vis_img.copy(), text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis_img


if __name__ == "__main__":
    results = []

    for pair_idx, cross_view_pair in enumerate(tqdm.tqdm(split_file["cross_view_pairs"])):
        cur_data = dset[all_uni_ids.index(cross_view_pair["model_id"])]
        src_view_idx, trg_view_idx = cross_view_pair["src_view"], cross_view_pair["trg_view"]

        cur_imgs, cur_poses, cur_focal = cur_data["images"], cur_data["poses"], cur_data["focal"]
        cur_masks = cur_data["masks"]
        cur_part_labels, cur_kp_labels = cur_data["part_labels"], cur_data["kp_labels"]

        src_part_label = cur_part_labels[src_view_idx]
        src_part_label_mask = src_part_label != 0
        src_unique_part_label = torch.unique(src_part_label).tolist()
        src_unique_part_label = sorted([x for x in src_unique_part_label if x != 0])
        src_kp_label = cur_kp_labels[src_view_idx]
        src_unique_kp_label = sorted(torch.unique(src_kp_label[..., -1]).tolist())

        trg_part_label = cur_part_labels[trg_view_idx]
        trg_part_label_mask = trg_part_label != 0
        trg_unique_part_label = torch.unique(trg_part_label).tolist()
        trg_unique_part_label = sorted([x for x in trg_unique_part_label if x != 0])
        trg_kp_label = cur_kp_labels[trg_view_idx]
        trg_unique_kp_label = sorted(torch.unique(trg_kp_label[..., -1]).tolist())
        inter_kp_labels = sorted(list(set(src_unique_kp_label) & set(trg_unique_kp_label)))

        NV, _, H, W = cur_imgs.shape

        grid = torch.stack(torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing="ij"), dim=-1)
        grid = grid.reshape(-1, 2).to(device=DEVICE)

        # ----------------- Extract feature map -----------------
        src_norm_feats = {}
        for method_name in model_info:
            src_norm_feats[method_name], _ = extract_feature_map(
                cur_imgs[src_view_idx : src_view_idx + 1],
                cur_poses[src_view_idx : src_view_idx + 1],
                cur_poses[src_view_idx : src_view_idx + 1],
                cur_focal,
                net=nets[method_name],
                render_par=render_pars[method_name],
                eval_ray_num=10000 if "diff" in method_name else 20000,
            )

        trg_norm_feats = {}
        trg_rgbs = {}
        for method_name in model_info:
            trg_norm_feats[method_name], trg_rgbs[method_name] = extract_feature_map(
                cur_imgs[src_view_idx : src_view_idx + 1],
                cur_poses[src_view_idx : src_view_idx + 1],
                cur_poses[trg_view_idx : trg_view_idx + 1],
                cur_focal,
                net=nets[method_name],
                render_par=render_pars[method_name],
                eval_ray_num=10000 if "diff" in method_name else 20000,
            )

        result = {**cross_view_pair}
        if args.save_vis:
            first_row, second_row = [], []
            src_img = cur_imgs[src_view_idx].cpu().numpy().transpose(1, 2, 0)
            src_mask = cur_masks[src_view_idx, 0].cpu().numpy()
            src_img = ((src_img + 1) * 0.5 * 255).astype(np.uint8)
            src_img[src_mask == 0] = [255, 255, 255]  # white background
            src_img_w_t = cv2.putText(src_img.copy(), "src", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            first_row.append(src_img_w_t)
            first_row.append(seg_map_to_vis_map(src_part_label.cpu().numpy(), "src_label"))
            second_row.append(draw_kps(src_img_w_t, src_kp_label, inter_kp_labels))

            trg_img = cur_imgs[trg_view_idx].cpu().numpy().transpose(1, 2, 0)
            trg_mask = cur_masks[trg_view_idx, 0].cpu().numpy()
            trg_img = ((trg_img + 1) * 0.5 * 255).astype(np.uint8).copy()
            trg_img[trg_mask == 0] = [255, 255, 255]  # white background
            trg_img_w_t = cv2.putText(trg_img.copy(), "trg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            first_row.append(trg_img_w_t)
            first_row.append(seg_map_to_vis_map(trg_part_label.cpu().numpy(), "trg_label"))
            second_row.append(draw_kps(trg_img_w_t, trg_kp_label, inter_kp_labels))

        for feat_type in src_norm_feats:
            src_norm_feat, trg_norm_feat = src_norm_feats[feat_type], trg_norm_feats[feat_type]
            src_kp_norm_feat = src_norm_feat.reshape(H, W, -1)[src_kp_label[:, 1], src_kp_label[:, 0]]
            src_valid_norm_feat = src_norm_feat[src_part_label_mask.reshape(-1)]
            trg_valid_norm_feat = trg_norm_feat[trg_part_label_mask.reshape(-1)]
            trg_valid_grid = grid[trg_part_label_mask.reshape(-1)]

            # ----------------- Novel view  -----------------
            trg_gt_img = cur_imgs[trg_view_idx].cpu().numpy().transpose(1, 2, 0)
            trg_gt_img = ((trg_gt_img + 1) * 0.5 * 255).astype(np.uint8).copy()
            trg_pred_img = trg_rgbs[feat_type].cpu().numpy().reshape(H, W, 3)
            trg_pred_img = (trg_pred_img * 255).astype(np.uint8).copy()
            psnr = skimage.metrics.peak_signal_noise_ratio(trg_pred_img, trg_gt_img, data_range=255)

            # ----------------- Part/KP matching -----------------
            corr = torch.matmul(trg_valid_norm_feat, src_valid_norm_feat.T)
            max_indices = torch.argmax(corr, dim=1)
            pred_part_labels = src_part_label[src_part_label_mask][max_indices].cpu().numpy()
            gt_part_labels = trg_part_label[trg_part_label_mask].cpu().numpy()
            assert gt_part_labels.shape == pred_part_labels.shape

            inter_unique_part_labels = sorted(list(set(src_unique_part_label) & set(trg_unique_part_label)))
            assert 0 not in inter_unique_part_labels
            part_ious = [0.0] * len(inter_unique_part_labels)
            for i, label in enumerate(inter_unique_part_labels):
                part_ious[i] = np.sum((gt_part_labels == label) & (pred_part_labels == label)) / float(
                    np.sum((gt_part_labels == label) | (pred_part_labels == label))
                )
            iou = np.mean(part_ious)

            kp_corr = torch.matmul(src_kp_norm_feat, trg_valid_norm_feat.T)
            kp_max_indices = torch.argmax(kp_corr, dim=1)
            pred_xy = trg_valid_grid[kp_max_indices]
            pred_uv = pred_xy[:, [1, 0]].cpu()

            pred_kp = torch.cat([pred_uv, src_kp_label[:, -1].reshape(-1, 1)], dim=-1)
            pred_kp = torch.cat([pred_kp[pred_kp[:, -1] == x] for x in inter_kp_labels])
            gt_kp = torch.cat([trg_kp_label[trg_kp_label[..., -1] == x] for x in inter_kp_labels])
            dist = np.linalg.norm(pred_kp[:, :2] - gt_kp[:, :2], axis=1)
            kp_accs = [np.sum(dist < thre) / float(len(dist)) for thre in [2.5, 5, 7.5, 10]]

            result[feat_type] = {
                "iou": iou,
                "psnr": psnr,
                "kp_accs": kp_accs,
                "pred_kp": pred_kp.numpy().tolist(),
                "gt_kp": gt_kp.numpy().tolist(),
            }

            if args.save_vis:
                seg_img = np.zeros((src_img.shape[0], src_img.shape[1]), dtype=np.uint8)
                seg_img[trg_part_label_mask] = pred_part_labels
                first_row.append(seg_map_to_vis_map(seg_img, feat_type))
                second_row.append(draw_kps(trg_img.copy(), pred_kp, inter_kp_labels, text=feat_type))
        results.append(result)

        if args.save_vis:
            first_row, second_row = np.concatenate(first_row, axis=1), np.concatenate(second_row, axis=1)
            second_row = np.pad(
                second_row,
                ((0, 0), (0, first_row.shape[1] - second_row.shape[1]), (0, 0)),
                "constant",
                constant_values=255,
            )
            vis_img = np.concatenate([first_row, second_row], axis=0)
            cv2.imwrite(os.path.join(output_dir, f"{pair_idx:06d}_vis.png"), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f)

    print("====================================")
    ious, kp_accs = {}, {}
    feat_types = [x for x in results[0] if isinstance(results[0][x], dict) and "iou" in results[0][x]]
    for feat_type in feat_types:
        ious[feat_type] = [x[feat_type]["iou"] for x in results]
        kp_accs[feat_type] = [x[feat_type]["kp_accs"] for x in results]
        print(f"{feat_type} iou: {np.mean(ious[feat_type]):.4f}")
        print(f"{feat_type} kp_accs: {np.round(np.mean(kp_accs[feat_type], axis=0), 4)}")

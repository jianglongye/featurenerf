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
import torch
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
output_dir = os.path.join(args.output.strip(), f"eval_feature_3d_all_{args.synset}")
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

src_ids = [x["src"] for x in split_file["cross_inst_pairs"]]
trg_ids = [x["trg"] for x in split_file["cross_inst_pairs"]]
all_uni_ids = sorted(list(set(src_ids + trg_ids)))
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
def extract_feature_vectors(img, pose, focal, pts, view_dirs, feat=None, net=None, H=128, W=128, eval_ray_num=10000):
    if feat is None:
        net.encode(img.to(device=DEVICE), pose.to(device=DEVICE), focal.to(device=DEVICE))

        output, feat = net(pts.to(DEVICE), viewdirs=view_dirs.to(DEVICE), ret_last_feat=True)
    else:
        raise NotImplementedError

    norm_feat = feat / torch.maximum(feat.norm(dim=-1, keepdim=True), torch.tensor(1e-6).to(feat.device))
    norm_feat = norm_feat.reshape(-1, norm_feat.shape[-1])

    return norm_feat


def seg_map_to_vis_map(img, uvs, seg_labels, text=None):
    def _hex_to_rgb(hex):
        return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))

    palette = ["70d6ff", "ff70a6", "ff9770", "ffd670", "e9ff70"]
    palette = [_hex_to_rgb(x) for x in palette]

    unique_seg_labels = np.unique(seg_labels)

    vis_img = np.zeros_like(img)
    palette_idx = 0
    for seg_label in unique_seg_labels:
        if seg_label == 0:
            continue
        mask = seg_labels == seg_label
        temp_uvs = uvs[mask]
        vis_img[temp_uvs[:, 1], temp_uvs[:, 0]] = palette[palette_idx % len(palette)]
        palette_idx += 1

    if text is not None:
        vis_img = cv2.putText(vis_img.copy(), text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis_img


def draw_kps(img, kps, kp_labels, text=None):
    colorpalette = sns.color_palette(n_colors=len(kp_labels))

    vis_img = img.copy()
    for kp_idx in range(kps.shape[0]):
        x, y, kp_id = kps[kp_idx].tolist()
        if kp_id not in kp_labels:
            continue
        color = colorpalette[kp_labels.index(kp_id)]
        color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        boundary_color = (color[0] // 2, color[1] // 2, color[2] // 2)
        # vis_img = cv2.circle(vis_img, (int(x), int(y)), 3, boundary_color, 1, cv2.LINE_AA)
        vis_img = cv2.circle(vis_img, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)

    if text is not None:
        vis_img = cv2.putText(vis_img.copy(), text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis_img


def rotate(pts: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
    new_pts = torch.matmul(pts, torch.transpose(rot_mat, -2, -1))
    return new_pts


def inverse(rot_or_tf_mat):
    if rot_or_tf_mat.shape[-1] == 3:
        new_mat = torch.transpose(rot_or_tf_mat, -2, -1)
    else:
        new_mat = rot_or_tf_mat.clone()
        new_mat[..., :3, :3] = torch.transpose(rot_or_tf_mat[..., :3, :3], -2, -1)
        if new_mat.ndim == 2:
            # reuse transpose rot here
            new_mat[:3, 3] = -rotate(new_mat[:3, 3], new_mat[:3, :3])
        else:
            new_mat[..., :3, 3] = -rotate(new_mat[..., None, :3, 3], new_mat[..., :3, :3]).squeeze(-2)
    return new_mat


def rot_tl_to_tf_mat(rot_mat: torch.Tensor, tl: torch.Tensor = None) -> torch.Tensor:
    tf_mat = torch.eye(4, device=rot_mat.device, dtype=rot_mat.dtype).repeat(rot_mat.shape[:-2] + (1, 1))
    tf_mat[..., :3, :3] = rot_mat
    if tl is not None:
        tf_mat[..., :3, 3] = tl
    return tf_mat


def project(pts, intr_mat):
    if intr_mat.ndim == 2 and pts.ndim == 2:
        pts = pts.clone()
        pts = pts / pts[:, 2:3]
        new_pts = torch.mm(pts, torch.transpose(intr_mat, 0, 1))
        return new_pts[:, :2]
    elif intr_mat.ndim == 3 and pts.ndim == 3:
        pts = pts.clone()
        pts = pts / pts[..., 2:3]
        new_pts = torch.bmm(pts, torch.transpose(intr_mat, 1, 2))
        return new_pts[..., :2]
    else:
        raise RuntimeError(f"Incorrect size of intr_mat or pts: {intr_mat.shape}, {pts.shape}")


def transform(pts: torch.Tensor, tf_mat: torch.Tensor) -> torch.Tensor:
    padding = torch.ones(pts.shape[:-1] + (1,), dtype=pts.dtype, device=pts.device)
    homo_pts = torch.cat([pts, padding], dim=-1)
    new_pts = torch.matmul(homo_pts, torch.transpose(tf_mat, -2, -1))
    new_pts = new_pts[..., :3]
    return new_pts


def compose_intr_mat(fu, fv, cu, cv, skew=0):
    intr_mat = np.array([[fu, skew, cu], [0, fv, cv], [0, 0, 1]])
    return intr_mat


def project_points(batch_poses: torch.Tensor, batch_points: torch.Tensor, focal, height, width):
    batch_size = batch_poses.shape[0]
    wld2opengl = inverse(batch_poses)
    opengl2opencv = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    wld2opencv = opengl2opencv[None].repeat(batch_size, 1, 1) @ wld2opengl

    intr = compose_intr_mat(focal, focal, width / 2, height / 2)
    intr = torch.from_numpy(intr)[None].repeat(batch_size, 1, 1).float()

    uvs = project(transform(batch_points, wld2opencv), intr).long()
    uvs[:, :, 0] = torch.clamp(uvs[:, :, 0], 0, width - 1)
    uvs[:, :, 1] = torch.clamp(uvs[:, :, 1], 0, height - 1)
    return uvs  # it's uv instead of xy


if __name__ == "__main__":
    results = []

    for pair_idx, cross_inst_pair in enumerate(tqdm.tqdm(split_file["cross_inst_pairs"])):
        src_data = dset[all_uni_ids.index(cross_inst_pair["src"])]
        trg_data = dset[all_uni_ids.index(cross_inst_pair["trg"])]
        view_idx = cross_inst_pair["view"]

        src_imgs, src_poses, src_focal = src_data["images"], src_data["poses"], src_data["focal"]
        trg_imgs, trg_poses, trg_focal = trg_data["images"], trg_data["poses"], trg_data["focal"]
        src_masks, trg_masks = src_data["masks"], trg_data["masks"]
        src_part_pts, src_part_labels = src_data["part_pts"], src_data["part_labels"]
        trg_part_pts, trg_part_labels = trg_data["part_pts"], trg_data["part_labels"]
        src_kp_labels, trg_kp_labels = src_data["kp_labels"], trg_data["kp_labels"]
        src_kp_pts = src_kp_labels[..., :3]
        trg_kp_pts = trg_data["kp_pts"]

        src_unique_part_label = torch.unique(src_part_labels).tolist()
        src_unique_part_label = sorted([x for x in src_unique_part_label if x != 0])
        src_unique_kp_label = sorted(torch.unique(src_kp_labels[..., -1]).tolist())
        trg_unique_part_label = torch.unique(trg_part_labels).tolist()
        trg_unique_part_label = sorted([x for x in trg_unique_part_label if x != 0])
        trg_unique_kp_label = sorted(torch.unique(trg_kp_labels[..., -1]).tolist())
        inter_kp_labels = sorted(list(set(src_unique_kp_label) & set(trg_unique_kp_label)))

        NV, _, H, W = src_imgs.shape

        src_pose, trg_pose = src_poses[view_idx], trg_poses[view_idx]

        src_grid_rays = util.gen_rays(src_pose.reshape(-1, 4, 4), W, H, src_focal, z_near, z_far)
        trg_grid_rays = util.gen_rays(trg_pose.reshape(-1, 4, 4), W, H, trg_focal, z_near, z_far)

        src_part_uvs = project_points(src_pose.reshape(-1, 4, 4), src_part_pts[None], src_focal, H, W)
        src_part_rays = src_grid_rays[0:1, src_part_uvs[0, :, 1], src_part_uvs[0, :, 0]]
        src_part_viewdirs = src_part_rays[..., 3:6]
        src_kp_uvs = project_points(src_pose.reshape(-1, 4, 4), src_kp_pts[None], src_focal, H, W)
        src_kp_rays = src_grid_rays[0:1, src_kp_uvs[0, :, 1], src_kp_uvs[0, :, 0]]
        src_kp_viewdirs = src_kp_rays[..., 3:6]

        trg_part_uvs = project_points(trg_pose.reshape(-1, 4, 4), trg_part_pts[None], trg_focal, H, W)
        trg_part_rays = trg_grid_rays[0:1, trg_part_uvs[0, :, 1], trg_part_uvs[0, :, 0]]
        trg_part_viewdirs = trg_part_rays[..., 3:6]
        trg_kp_uvs = project_points(trg_pose.reshape(-1, 4, 4), trg_kp_pts[None], trg_focal, H, W)
        trg_kp_rays = trg_grid_rays[0:1, trg_kp_uvs[0, :, 1], trg_kp_uvs[0, :, 0]]
        trg_kp_viewdirs = trg_kp_rays[..., 3:6]

        # ----------------- Extract feature vectors -----------------
        src_part_norm_feats, src_kp_norm_feats = {}, {}
        for method_name in model_info:
            src_part_norm_feats[method_name] = extract_feature_vectors(
                src_imgs[view_idx : view_idx + 1],
                src_poses[view_idx : view_idx + 1],
                src_focal,
                src_part_pts[None],
                src_part_viewdirs,
                feat=None,
                net=nets[method_name],
            )
            src_kp_norm_feats[method_name] = extract_feature_vectors(
                src_imgs[view_idx : view_idx + 1],
                src_poses[view_idx : view_idx + 1],
                src_focal,
                src_kp_pts[None],
                src_kp_viewdirs,
                feat=None,
                net=nets[method_name],
            )

        trg_part_norm_feats, trg_kp_norm_feats = {}, {}
        for method_name in model_info:
            trg_part_norm_feats[method_name] = extract_feature_vectors(
                trg_imgs[view_idx : view_idx + 1],
                trg_poses[view_idx : view_idx + 1],
                trg_focal,
                trg_part_pts[None],
                trg_part_viewdirs,
                feat=None,
                net=nets[method_name],
            )
            trg_kp_norm_feats[method_name] = extract_feature_vectors(
                trg_imgs[view_idx : view_idx + 1],
                trg_poses[view_idx : view_idx + 1],
                trg_focal,
                trg_kp_pts[None],
                trg_kp_viewdirs,
                feat=None,
                net=nets[method_name],
            )

        result = {**cross_inst_pair}
        result["src_kp"] = src_kp_labels.cpu().numpy().tolist()
        if args.save_vis:
            first_row, second_row = [], []
            src_img = src_imgs[view_idx].cpu().numpy().transpose(1, 2, 0)
            src_mask = src_masks[view_idx, 0].cpu().numpy()
            src_img = ((src_img + 1) * 0.5 * 255).astype(np.uint8)
            src_img[src_mask == 0] = [255, 255, 255]  # white background
            src_img_w_t = cv2.putText(src_img.copy(), "src", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            first_row.append(src_img_w_t)
            src_seg_img = seg_map_to_vis_map(src_img, src_part_uvs[0], src_part_labels, "src_label")
            first_row.append(src_seg_img)
            src_gt_kp_uvs = project_points(src_pose.reshape(-1, 4, 4), src_kp_labels[None, :, :3], src_focal, H, W)
            src_gt_kp = np.concatenate([src_gt_kp_uvs[0], src_kp_labels[:, 3:]], axis=1)
            src_kp_img = draw_kps(src_img, src_gt_kp, inter_kp_labels, "src_kp")
            second_row.append(src_kp_img)

            trg_img = trg_imgs[view_idx].cpu().numpy().transpose(1, 2, 0)
            trg_mask = trg_masks[view_idx, 0].cpu().numpy()
            trg_img = ((trg_img + 1) * 0.5 * 255).astype(np.uint8).copy()
            trg_img[trg_mask == 0] = [255, 255, 255]  # white background
            trg_img_w_t = cv2.putText(trg_img.copy(), "trg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            first_row.append(trg_img_w_t)
            trg_seg_img = seg_map_to_vis_map(trg_img, trg_part_uvs[0], trg_part_labels, "trg_label")
            first_row.append(trg_seg_img)
            trg_gt_kp_uvs = project_points(trg_pose.reshape(-1, 4, 4), trg_kp_labels[None, :, :3], trg_focal, H, W)
            trg_gt_kp = np.concatenate([trg_gt_kp_uvs[0], trg_kp_labels[:, 3:]], axis=1)
            trg_kp_img = draw_kps(trg_img, trg_gt_kp, inter_kp_labels, "trg_kp")
            second_row.append(trg_kp_img)

            os.makedirs(os.path.join(output_dir, "individual_files", f"{pair_idx:06d}"), exist_ok=True)
            np.savez_compressed(
                os.path.join(output_dir, "individual_files", f"{pair_idx:06d}", "labels.npz"),
                src_part_labels=src_part_labels.cpu().numpy(),
                src_part_pts=src_part_pts.cpu().numpy(),
                trg_part_labels=trg_part_labels.cpu().numpy(),
                trg_part_pts=trg_part_pts.cpu().numpy(),
            )

        for feat_type in src_part_norm_feats:
            src_part_norm_feat, trg_part_norm_feat = src_part_norm_feats[feat_type], trg_part_norm_feats[feat_type]
            src_kp_norm_feat, trg_kp_norm_feat = src_kp_norm_feats[feat_type], trg_kp_norm_feats[feat_type]

            # ----------------- Part/KP matching -----------------
            corr = torch.matmul(trg_part_norm_feat, src_part_norm_feat.T)
            max_indices = torch.argmax(corr, dim=1)
            pred_part_labels = src_part_labels[max_indices].cpu().numpy()
            gt_part_labels = trg_part_labels.cpu().numpy()
            assert gt_part_labels.shape == pred_part_labels.shape

            inter_unique_part_labels = sorted(list(set(src_unique_part_label) & set(trg_unique_part_label)))
            assert 0 not in inter_unique_part_labels
            part_ious = [0.0] * len(inter_unique_part_labels)
            for i, label in enumerate(inter_unique_part_labels):
                part_ious[i] = np.sum((gt_part_labels == label) & (pred_part_labels == label)) / float(
                    np.sum((gt_part_labels == label) | (pred_part_labels == label))
                )
            iou = np.mean(part_ious)

            kp_corr = torch.matmul(src_kp_norm_feat, trg_kp_norm_feat.T)
            kp_max_indices = torch.argmax(kp_corr, dim=1)
            pred_pts = trg_kp_pts[kp_max_indices]

            pred_kp = torch.cat([pred_pts, src_kp_labels[:, -1].reshape(-1, 1)], dim=-1)
            pred_kp = torch.cat([pred_kp[pred_kp[:, -1] == x] for x in inter_kp_labels])
            gt_kp = torch.cat([trg_kp_labels[trg_kp_labels[..., -1] == x] for x in inter_kp_labels])
            dist = np.linalg.norm(pred_kp[:, :3] - gt_kp[:, :3], axis=1)
            kp_accs = [np.sum(dist < thre) / float(len(dist)) for thre in [0.025, 0.05, 0.075, 0.1]]

            result[feat_type] = {
                "iou": iou,
                "kp_accs": kp_accs,
                "pred_kp": pred_kp.numpy().tolist(),
                "gt_kp": gt_kp.numpy().tolist(),
            }

            if args.save_vis:
                pred_seg_img = seg_map_to_vis_map(trg_img, trg_part_uvs[0], pred_part_labels, feat_type)
                first_row.append(pred_seg_img)
                pred_kp_uvs = project_points(trg_pose.reshape(-1, 4, 4), pred_kp[None, :, :3], trg_focal, H, W)
                second_row.append(
                    draw_kps(
                        trg_img.copy(),
                        np.concatenate([pred_kp_uvs[0], pred_kp[:, 3:]], axis=1),
                        inter_kp_labels,
                        text=feat_type,
                    )
                )
                np.savez_compressed(
                    os.path.join(output_dir, "individual_files", f"{pair_idx:06d}", f"{feat_type}.npz"),
                    pred_part_labels=pred_part_labels,
                )

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
            cv2.imwrite(os.path.join(output_dir, f"{pair_idx:06d}.png"), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

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

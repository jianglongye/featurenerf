# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import trainlib
from dotmap import DotMap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import util
from data import get_split_dataset
from model import loss, make_model
from render import NeRFEmbedRenderer


def extra_args(parser):
    parser.add_argument("--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')")
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument("--no_wandb", action="store_true")
    return parser


args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
device = util.get_cuda(args.gpu_id[0])

if args.dataset_format.startswith("feat"):
    extra_dataset_kwargs = {"synset": args.synset, "white_bkgd": conf["renderer"].get_bool("white_bkgd")}
elif args.dataset_format.startswith("co3d"):
    extra_dataset_kwargs = {"category": args.category}
else:
    extra_dataset_kwargs = {}

dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir, **extra_dataset_kwargs)
print("dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp))

net = make_model(conf["model"]).to(device=device)
net.stop_encoder_grad = args.freeze_enc
if args.freeze_enc:
    print("Encoder frozen")
    net.encoder.eval()

regress_coord = conf.get_float("loss.lambda_coord", 0.0) > 0
renderer = NeRFEmbedRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp, regress_coord=regress_coord).to(
    device=device
)

# Parallize
render_par = renderer.bind_parallel(net, args.gpu_id).eval()

nviews = list(map(int, args.nviews.split()))

util.set_random_seed(42)


class FeatNeRFTrainer(trainlib.TrainerWandb):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        self.lambda_embed = conf.get_float("loss.lambda_embed", 1.0)
        self.lambda_coord = conf.get_float("loss.lambda_coord", 0.0)
        print(
            "lambda coarse {}, fine {}, embed {}, coord {}".format(
                self.lambda_coarse, self.lambda_fine, self.lambda_embed, self.lambda_coord
            )
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        self.embed_crit = torch.nn.MSELoss()
        self.coord_crit = torch.nn.MSELoss()

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(torch.load(self.renderer_state_path, map_location=device))

        self.z_near = dset.z_near
        self.z_far = dset.z_far

        self.use_bbox = args.no_bbox_step > 0
        self.mask_feat = conf.get_bool("data.mask_feat", False)
        self.mask_white_bkgd = conf.get_bool("renderer.white_bkgd", True)

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = data["focal"]  # (SB)
        all_c = data.get("c")  # (SB)
        if is_train:
            all_feats = data["feats"].to(device=device)  # (SB, NV, D, H // 8, W // 8)

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)

        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_feat_gt = []
        all_rays = []

        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            if is_train:
                feats = all_feats[obj_idx]  # (NV, D, H // 8, W // 8)
            c = None
            if "c" in data:
                c = data["c"][obj_idx]
            if curr_nviews > 1:
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(np.random.choice(NV, curr_nviews, replace=False))
            images_0to1 = images * 0.5 + 0.5

            cam_rays = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c)  # (NV, H, W, 8)
            rgb_gt_all = images_0to1
            rgb_gt_all = rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)  # (NV, H, W, 3)

            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, args.ray_batch_size)
                # pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
            else:
                pix_0 = torch.randint(0, NV, (args.ray_batch_size,))
                pix_1 = torch.randint(0, H, (args.ray_batch_size,))
                pix_2 = torch.randint(0, W, (args.ray_batch_size,))
                pix = torch.stack([pix_0, pix_1, pix_2], dim=-1)
                # pix_inds = torch.randint(0, NV * H * W, (args.ray_batch_size,))
            pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]

            rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3)
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(device=device)  # (ray_batch_size, 8)

            if is_train:
                grid = torch.flip(pix[..., 1:].clone(), dims=(-1,))  # !!! note the index order of the grid sample
                grid_n1to1 = (grid.float() / torch.tensor([H, W]).float()) * 2.0 - 1.0
                grid_n1to1 = grid_n1to1[None, None].repeat(NV, 1, 1, 1)
                feat_gt = F.grid_sample(feats, grid_n1to1.to(device), align_corners=False)
                feat_gt = feat_gt[pix[..., 0], :, 0, torch.arange(pix.shape[0])]
                all_feat_gt.append(feat_gt)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)
        if is_train:
            all_feat_gt = torch.stack(all_feat_gt)  # (SB, ray_batch_size, D)
            if self.mask_feat:
                if self.mask_white_bkgd:
                    all_mask_gt = 1 - torch.all((all_rgb_gt == 1.0), dim=-1).float()
                else:
                    all_mask_gt = 1 - torch.all((all_rgb_gt == 0.0), dim=-1).float()
                all_feat_gt = all_feat_gt * all_mask_gt[..., None]

        image_ord = image_ord.to(device)
        src_images = util.batched_index_select_nd(all_images, image_ord)  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        all_bboxes = all_poses = all_images = None

        if len(all_focals.shape) > 1 and all_focals.shape[-1] > 1:
            assert image_ord.shape[-1] == 1, "not implemented"
            src_focals = util.batched_index_select_nd(all_focals.to(device), image_ord).squeeze(-1)
            src_c = util.batched_index_select_nd(all_c.to(device), image_ord).squeeze(1)
            net.encode(src_images, src_poses, src_focals, c=src_c)
        else:
            net.encode(
                src_images,
                src_poses,
                all_focals.to(device=device),
                c=all_c.to(device=device) if all_c is not None else None,
            )

        render_dict = DotMap(
            render_par(
                all_rays,
                want_weights=True,
            )
        )
        coarse = render_dict.coarse
        fine = render_dict.fine
        using_fine = len(fine) > 0

        loss_dict = {}

        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
        if is_train:
            if self.lambda_coord > 0:
                coord_loss = self.coord_crit(coarse.coord, torch.zeros_like(coarse.coord))
                loss_dict["cc"] = coord_loss.item() * self.lambda_coord
            embed_loss = self.embed_crit(coarse.embed, all_feat_gt)
            loss_dict["ec"] = embed_loss.item() * self.lambda_embed
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine
            if is_train:
                if self.lambda_coord > 0:
                    fine_coord_loss = self.coord_crit(fine.coord, torch.zeros_like(fine.coord))
                    loss_dict["cf"] = fine_coord_loss.item() * self.lambda_coord
                fine_embed_loss = self.embed_crit(fine.embed, all_feat_gt)
                embed_loss = (embed_loss + fine_embed_loss) * self.lambda_embed
                loss_dict["ef"] = fine_embed_loss.item() * self.lambda_embed

                if self.lambda_coord > 0:
                    coord_loss = (coord_loss + fine_coord_loss) * self.lambda_coord

        loss = rgb_loss
        if is_train:
            loss = loss + embed_loss
            if self.lambda_coord > 0:
                loss = loss + coord_loss
            loss.backward()
        loss_dict["t"] = loss.item()

        return loss_dict

    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx : batch_idx + 1]  # (1)
        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c)  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        view_dest = np.random.randint(0, NV - curr_nviews)
        for vs in range(curr_nviews):
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        # set renderer net to eval mode
        renderer.eval()
        source_views = images_0to1[views_src].permute(0, 2, 3, 1).cpu().numpy().reshape(-1, H, W, 3)

        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H, W, 8)
            test_images = images[views_src]  # (NS, 3, H, W)
            if len(focal.shape) > 1 and focal.shape[-1] > 1:
                net.encode(
                    test_images.unsqueeze(0),
                    poses[views_src].unsqueeze(0),
                    focal[..., views_src].squeeze(1).to(device=device),
                    c=c[..., views_src, :].squeeze(1).to(device=device) if c is not None else None,
                )
            else:
                net.encode(
                    test_images.unsqueeze(0),
                    poses[views_src].unsqueeze(0),
                    focal.to(device=device),
                    c=c.to(device=device) if c is not None else None,
                )

            test_rays = test_rays.reshape(1, H * W, -1)
            alpha_coarse_np, rgb_coarse_np, depth_coarse_np, embed_coarse_np = [], [], [], []
            alpha_fine_np, rgb_fine_np, depth_fine_np, embed_fine_np = [], [], [], []
            for _temp_rays in torch.split(test_rays, 10000, dim=1):
                _temp_dict = DotMap(render_par(_temp_rays, want_weights=True))

                # render_dict = DotMap(render_par(test_rays, want_weights=True))
                coarse = _temp_dict.coarse
                fine = _temp_dict.fine
                using_fine = len(fine) > 0
                alpha_coarse_np.append(coarse.weights[0].sum(dim=-1).cpu().numpy())
                rgb_coarse_np.append(coarse.rgb[0].cpu().numpy())
                depth_coarse_np.append(coarse.depth[0].cpu().numpy())
                embed_coarse_np.append(coarse.embed[0].cpu().numpy())

                if using_fine:
                    alpha_fine_np.append(fine.weights[0].sum(dim=1).cpu().numpy())
                    rgb_fine_np.append(fine.rgb[0].cpu().numpy())
                    depth_fine_np.append(fine.depth[0].cpu().numpy())
                    embed_fine_np.append(fine.embed[0].cpu().numpy())

            alpha_coarse_np = np.concatenate(alpha_coarse_np, axis=0).reshape(H, W)
            rgb_coarse_np = np.concatenate(rgb_coarse_np, axis=0).reshape(H, W, 3)
            depth_coarse_np = np.concatenate(depth_coarse_np, axis=0).reshape(H, W)
            embed_coarse_np = np.concatenate(embed_coarse_np, axis=0).reshape(H, W, -1).mean(-1)
            if using_fine:
                alpha_fine_np = np.concatenate(alpha_fine_np, axis=0).reshape(H, W)
                rgb_fine_np = np.concatenate(rgb_fine_np, axis=0).reshape(H, W, 3)
                depth_fine_np = np.concatenate(depth_fine_np, axis=0).reshape(H, W)
                embed_fine_np = np.concatenate(embed_fine_np, axis=0).reshape(H, W, -1).mean(-1)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print("c alpha min {}, max {}".format(alpha_coarse_np.min(), alpha_coarse_np.max()))
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        embed_coarse_cmap = util.cmap(embed_coarse_np) / 255
        vis_list = [
            *source_views,
            gt,
            depth_coarse_cmap,
            rgb_coarse_np,
            alpha_coarse_cmap,
            embed_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print("f alpha min {}, max {}".format(alpha_fine_np.min(), alpha_fine_np.max()))
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            embed_fine_cmap = util.cmap(embed_fine_np) / 255
            vis_list = [
                *source_views,
                gt,
                depth_fine_cmap,
                rgb_fine_np,
                alpha_fine_cmap,
                embed_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        renderer.train()
        return vis, vals


trainer = FeatNeRFTrainer()
trainer.start()

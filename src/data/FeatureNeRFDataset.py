import glob
import json
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from util import get_image_to_tensor_balanced, get_mask_to_tensor

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets")


def fov_to_focal_length(fov, image_or_sensor_size):
    return image_or_sensor_size / (2 * np.tan(fov / 2))


class FeatureNeRFDataset(Dataset):
    def __init__(
        self,
        data_root,
        synset="03001627",
        stage="train",
        image_size=(128, 128),
        world_scale=1.0,
        use_feat=False,
        feat_type="dino_128",
        use_part_anno=False,
        part_anno_type="part_16_anno",
        use_kp_anno=False,
        kp_anno_type="kp_2d_anno",
        model_ids=None,
        target_views=(-1,),
        white_bkgd=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.synset = synset
        assert os.path.exists(self.data_root), "Data root does not exist: " + self.data_root
        print("Loading feature nerf dataset", "synset:", self.synset)

        self.camera_dir = os.path.join(data_root, "camera", synset)
        self.img_dir = os.path.join(data_root, "img", synset)
        self.stage = stage

        self.use_feat = use_feat
        self.feat_type = feat_type
        self.use_part_anno = use_part_anno
        self.part_ano_type = part_anno_type
        self.use_2d_part_anno = "2d" in part_anno_type
        self.use_kp_anno = use_kp_anno
        self.use_2d_kp_anno = "2d" in kp_anno_type
        self.kp_anno_type = kp_anno_type

        self.target_views = target_views
        self.use_all_view = -1 in target_views

        self.feat_dir = os.path.join(data_root, feat_type, synset)

        split_path = os.path.join(ASSETS_DIR, "splits", synset + ".json")
        with open(split_path, "r") as f:
            split_file = json.load(f)
        self.obj_ids = split_file[stage]
        # remove invalid obj ids
        self.obj_ids = [x for x in self.obj_ids if x != "f9c1d7748c15499c6f2bd1c4e9adb41"]

        if use_part_anno:
            self.part_anno_dir = os.path.join(data_root, part_anno_type, synset)
            part_anno_paths = sorted(glob.glob(os.path.join(self.part_anno_dir, "*.npz")))
            part_anno_ids = [os.path.basename(x).split(".")[0] for x in part_anno_paths]
            self.obj_ids = [x for x in self.obj_ids if x in part_anno_ids]
        if use_kp_anno:
            self.kp_anno_dir = os.path.join(data_root, kp_anno_type, synset)
            kp_anno_paths = sorted(glob.glob(os.path.join(self.kp_anno_dir, "*.npz")))
            kp_anno_ids = [os.path.basename(x).split(".")[0] for x in kp_anno_paths]
            self.obj_ids = [x for x in self.obj_ids if x in kp_anno_ids]
        if model_ids is not None:
            assert all(
                [x in self.obj_ids for x in model_ids]
            ), "Not all model ids are found in the dataset, please check the split file"
            self.obj_ids = model_ids
        print("Using {} objects for stage {}".format(len(self.obj_ids), stage))

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

        self.z_near = 0.8
        self.z_far = 1.8
        self.lindisp = False
        self.white_bkgd = white_bkgd

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        if ch == 3:
            return rgba

        assert ch == 4, "RGBA image has 4 channels."

        rgb = np.zeros((row, col, 3), dtype="float32")
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype="float32") / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype="uint8")

    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, index):
        obj_id = self.obj_ids[index]

        transforms_path = os.path.join(self.camera_dir, obj_id + ".json")
        with open(transforms_path, "r") as f:
            transforms = json.load(f)

        rgb_dir = os.path.join(self.img_dir, obj_id)
        rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        assert len(rgb_paths) == len(transforms["frames"])

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        for frame in transforms["frames"]:
            rgb_path = os.path.join(rgb_dir, frame["file_path"])

            img = imageio.imread(rgb_path)
            if self.white_bkgd:
                img_rgb = self.rgba2rgb(img, background=(255, 255, 255))
            else:
                img_rgb = self.rgba2rgb(img, background=(0, 0, 0))
            img_tensor = self.image_to_tensor(img_rgb)
            mask = img[..., 3:]
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(np.asarray(frame["transform_matrix"]).reshape(4, 4)).float()
            pose = pose @ self._coord_trans

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError("ERROR: Bad image at", rgb_path, "please investigate!")
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if self.use_feat and self.stage in ["train", "test"]:
            feat_path = os.path.join(self.feat_dir, obj_id + ".npz")
            assert os.path.exists(feat_path), "Feature path does not exist: " + feat_path
            feats = np.load(feat_path, allow_pickle=True)["feats"]
            if self.use_all_view:
                assert feats.shape[0] == len(transforms["frames"])
            else:
                if feats.dtype == np.object:
                    feats = feats.tolist()
                    assert isinstance(feats, dict) and all([view in feats for view in self.target_views])
                    temp_feats = np.zeros(
                        (len(rgb_paths), *feats[self.target_views[0]].shape), dtype=feats[self.target_views[0]].dtype
                    )
                    for view in self.target_views:
                        temp_feats[view] = feats[view]
                    feats = temp_feats
                else:
                    assert len(feats) == len(rgb_paths)
            all_feats = torch.from_numpy(feats).float()

        if self.use_part_anno:
            part_anno_path = os.path.join(self.part_anno_dir, obj_id + ".npz")
            assert os.path.exists(part_anno_path), "Missing part anno file: " + part_anno_path
            part_anno = np.load(part_anno_path)
            part_labels = part_anno["labels"]
            if not self.use_2d_part_anno:
                part_pts = part_anno["pts"]
                assert len(part_pts) == len(part_labels)
            else:
                assert len(part_labels) == len(rgb_paths)

        if self.use_kp_anno:
            kp_anno_path = os.path.join(self.kp_anno_dir, obj_id + ".npz")
            assert os.path.exists(kp_anno_path), "Missing kp anno file: " + kp_anno_path
            kp_anno = np.load(kp_anno_path, allow_pickle=True)
            kp_labels = kp_anno["labels"]
            if not self.use_2d_kp_anno:
                kp_pts = kp_anno["pts"]
            else:
                kp_labels = kp_labels.tolist()
                assert len(kp_labels) == len(rgb_paths)

        camera_angle = float(transforms["camera_angle_x"])
        focal = fov_to_focal_length(camera_angle, self.image_size[0])
        cx, cy = self.image_size[0] / 2, self.image_size[1] / 2

        if self.use_part_anno:
            if not self.use_2d_part_anno:
                part_pts = torch.from_numpy(part_pts).float()
            part_labels = torch.from_numpy(part_labels).long()
        if self.use_kp_anno:
            if not self.use_2d_kp_anno:
                kp_pts = torch.from_numpy(kp_pts).float()
                kp_labels = torch.from_numpy(kp_labels).float()
            else:
                kp_labels = [torch.from_numpy(kp_label).long() for kp_label in kp_labels]

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        result = {
            "path": obj_id,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        if self.use_feat and self.stage in ["train", "test"]:
            result["feats"] = all_feats
        if self.use_part_anno:
            if not self.use_2d_part_anno:
                result["part_pts"] = part_pts
            result["part_labels"] = part_labels
        if self.use_kp_anno:
            if not self.use_2d_kp_anno:
                result["kp_pts"] = kp_pts
            result["kp_labels"] = kp_labels
        return result

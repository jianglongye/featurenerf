from .FeatureNeRFDataset import FeatureNeRFDataset
from .SRNDataset import SRNDataset


def get_split_dataset(dataset_type, datadir, want_split="all", training=True, **kwargs):
    """
    Retrieved desired dataset class
    :param dataset_type dataset type name (srn|dvr|dvr_gen, etc)
    :param datadir root directory name for the dataset. For SRN/multi_obj data:
    if data is in dir/cars_train, dir/cars_test, ... then put dir/cars
    :param want_split root directory name for the dataset
    :param training set to False in eval scripts
    """
    dset_class, train_aug = None, None
    flags, train_aug_flags = {}, {}

    if dataset_type == "srn":
        # For ShapeNet single-category (from SRN)
        dset_class = SRNDataset
    elif dataset_type.startswith("feature"):
        dset_class = FeatureNeRFDataset
        assert "synset" in kwargs, "Must specify synset for feature nerf"
        if dataset_type == "feature":
            pass
        if "2d_part_kp_anno" in dataset_type:
            flags["use_part_anno"] = True
            flags["part_anno_type"] = "part_2d_anno"
            flags["use_kp_anno"] = True
            flags["kp_anno_type"] = "kp_2d_anno"
        elif "3d_part_kp_anno" in dataset_type:
            flags["use_part_anno"] = True
            flags["part_anno_type"] = "part_3d_anno"
            flags["use_kp_anno"] = True
            flags["kp_anno_type"] = "kp_3d_anno"

        if "dino_256" in dataset_type:
            flags["use_feat"] = True
            flags["feat_type"] = "dino_feat_256"
        elif "diff_512" in dataset_type:
            flags["use_feat"] = True
            flags["feat_type"] = "diff_feat_512"
        else:
            raise ValueError(f"Unknown dataset type {dataset_type}")
    else:
        raise ValueError("Unsupported dataset type", dataset_type)

    want_train = want_split != "val" and want_split != "test"
    want_val = want_split != "train" and want_split != "test"
    want_test = want_split != "train" and want_split != "val"

    for k in kwargs:
        if k in flags:
            flags.pop(k)

    if want_train:
        train_set = dset_class(datadir, stage="train", **flags, **kwargs)
        if train_aug is not None:
            train_set = train_aug(train_set, **train_aug_flags)

    if want_val:
        val_set = dset_class(datadir, stage="val", **flags, **kwargs)
    if want_test:
        test_set = dset_class(datadir, stage="test", **flags, **kwargs)

    if want_split == "train":
        return train_set
    elif want_split == "val":
        return val_set
    elif want_split == "test":
        return test_set
    return train_set, val_set, test_set

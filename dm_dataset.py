import glob
import h5py
import torch
import os
from torch.utils.data import Dataset

from dm_data_utils import *


class DeepMapDataset(Dataset):

    def __init__(
        self, pdb_list, data_root, normalize=False, transform=False,
        build_report=False, rot_label=True, verbose=False
    ):
        self.h5_files = []
        self.n = 0
        self._build(pdb_list, data_root, build_report=build_report)
        self.normalize = normalize
        self.transform = transform
        self.rot_label = rot_label
        self.verbose = verbose

    def _build(self, pdb_list, data_root, build_report=False):
        print("Building dataset")
        pdbs = [p.strip() for p in open(pdb_list).readlines()]
        for pdb in pdbs:
            meta_files = glob.glob(os.path.join(data_root, f"{pdb}", "*.map.h5"))
            print(pdb, len(meta_files))
            self.h5_files += meta_files
        self.n = len(self.h5_files)
        if build_report:
            print("Building file summary")
            df_report = []
            for pdb in pdbs:
                files = glob.glob(os.path.join(data_root, f"{pdb}", "*.map.h5"))
                for file_path in files:
                    with h5py.File(file_path, "r") as f:
                        x_offset = f.attrs["x_offset"]
                        y_offset = f.attrs["y_offset"]
                        z_offset = f.attrs["z_offset"]
                    df_report.append(
                        dict(
                            pdb=pdb,
                            x_offset=x_offset,
                            y_offset=y_offset,
                            z_offset=z_offset,
                            path=file_path
                        )
                    )
            df_report = pd.DataFrame(df_report)
            df_report.to_csv("predict/summary.csv")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        file_path = self.h5_files[idx]
        if self.verbose:
            print(f"[{idx}] {file_path}")
        with h5py.File(file_path.replace(".meta.", ".map."), "r") as f:
            box = f["data"][()]
            x_offset = f.attrs["x_offset"]
            y_offset = f.attrs["y_offset"]
            z_offset = f.attrs["z_offset"]
            vol_mean = f.attrs["mean"]
            vol_std = f.attrs["std"]
        if self.verbose:
            print("x_offset, y_offset, z_offset = {:.3f}, {:.3f}, {:.3f}"
                .format(x_offset, y_offset, y_offset))
        if self.normalize:
            box = normalize_data(box, mean=vol_mean, std=vol_std)
        label_coor, label_rot = build_labels(
            x_offset,
            y_offset,
            z_offset,
            file_path,
            use_rot_label=self.rot_label
        )
        box = torch.from_numpy(box).unsqueeze(0)
        label_coor = torch.from_numpy(label_coor)
        label_rot = torch.from_numpy(label_rot)
        if self.transform:
            box, label_coor, label_rot = transform_data(box, label_coor, label_rot)
        return box, label_coor, label_rot

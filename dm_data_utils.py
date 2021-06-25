import glob
import h5py
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from aa_code_utils import *


def normalize_data(x, mean=None, std=None, eps=0.001):
    if mean is None:
        mean = np.mean(x)
    if std is None:
        std = np.std(x)
    x = x - mean
    x = x / max(eps, float(std))
    return x


def build_labels(
        x_offset, y_offset, z_offset, file_path, label_dim=16, dx=0.25,
        downsample=8, use_rot_label=True, verbose=False
    ):
    csv_path = file_path.replace(".map.", ".meta.")  # .replace(".h5", ".csv")
    df = pd.read_hdf(csv_path, "df") # .drop_duplicates(subset=["chain", "res_no", "atom_name"])
    if verbose:
        print(df)
    df_ca = df.query("atom_name == 'CA'").rename(columns={"x": "ca_x", "y": "ca_y", "z": "ca_z"})
    df_c = df.query("atom_name == 'C'").rename(columns={"x": "c_x", "y": "c_y", "z": "c_z"})
    df_n = df.query("atom_name == 'N'").rename(columns={"x": "n_x", "y": "n_y", "z": "n_z"})
    df_o = df.query("atom_name == 'O'").rename(columns={"x": "o_x", "y": "o_y", "z": "o_z"})
    if verbose:
        print("df_ca", len(df_ca), "df_c", len(df_c), "df_n", len(df_n), "df_o", len(df_o))
    df_bb = df_ca.merge(df_c[["chain", "res_no", "c_x", "c_y", "c_z"]], on=["chain", "res_no"], how="inner")
    df_bb = df_bb.merge(df_n[["chain", "res_no", "n_x", "n_y", "n_z"]], on=["chain", "res_no"], how="inner")
    df_bb = df_bb.merge(df_o[["chain", "res_no", "o_x", "o_y", "o_z"]], on=["chain", "res_no"], how="inner")
    coor_labels = np.zeros((13, label_dim, label_dim, label_dim), dtype=np.float)
    rot_labels = np.zeros((1, label_dim, label_dim, label_dim), dtype=np.int)
    for i in range(len(df_bb)):
        ca_x, ca_y, ca_z, rot_label = df_bb.iloc[i][["ca_x", "ca_y", "ca_z", "rot_label"]]
        u = 1 / (downsample * dx) * (ca_z - z_offset)
        v = 1 / (downsample * dx) * (ca_y - y_offset)
        w = 1 / (downsample * dx) * (ca_x - x_offset)
        if u >= label_dim or v >= label_dim or w >= label_dim:
            continue
        # CA confidence
        coor_labels[0, int(u), int(v), int(w)] = 1
        # CA coors
        coor_labels[1, int(u), int(v), int(w)] = u - int(u)
        coor_labels[2, int(u), int(v), int(w)] = v - int(v)
        coor_labels[3, int(u), int(v), int(w)] = w - int(w)
        # C coors
        coor_labels[4:7, int(u), int(v), int(w)] = (
                df_bb.iloc[i][["c_z", "c_y", "c_x"]].values - df_bb.iloc[i][["ca_z", "ca_y", "ca_x"]].values
        )
        # N coors
        coor_labels[7:10, int(u), int(v), int(w)] = (
                df_bb.iloc[i][["n_z", "n_y", "n_x"]].values - df_bb.iloc[i][["ca_z", "ca_y", "ca_x"]].values
        )
        # O coors
        coor_labels[10:13, int(u), int(v), int(w)] = (
                df_bb.iloc[i][["o_z", "o_y", "o_x"]].values - df_bb.iloc[i][["ca_z", "ca_y", "ca_x"]].values
        )
        # Rotamer label
        if use_rot_label:
            rot_labels[0, int(u), int(v), int(w)] = rot_to_idx(rot_label)
        else:
            rot_labels[0, int(u), int(v), int(w)] = rot_to_aa_idx(rot_label)
    return coor_labels, rot_labels


def make_vector_map(coors, data_shape, downsample=1):
    output = np.zeros((4, *[int(np.ceil(i / downsample)) for i in data_shape]))
    for coor in coors:
        i = int(coor[0] / downsample)
        j = int(coor[1] / downsample)
        k = int(coor[2] / downsample)
        output[0, i, j, k] = 1
        output[1, i, j, k] = np.fmod(coor[0] / downsample, 1)
        output[2, i, j, k] = np.fmod(coor[1] / downsample, 1)
        output[3, i, j, k] = np.fmod(coor[2] / downsample, 1)
    return output


def make_field_map(coors, ca_coors, data_shape, downsample=1):
    output = np.zeros((3, *[int(np.ceil(i / downsample)) for i in data_shape]))
    for (coor, ca_coor) in zip(coors, ca_coors):
        i = int(ca_coor[0] / downsample)
        j = int(ca_coor[1] / downsample)
        k = int(ca_coor[2] / downsample)
        # dis = np.linalg.norm(coor - ca_coor)
        vec = (coor - ca_coor)
        output[0, i, j, k] = vec[0] / downsample  # coor[2] / downsample - i
        output[1, i, j, k] = vec[1] / downsample  # coor[1] / downsample - j
        output[2, i, j, k] = vec[2] / downsample  # coor[0] / downsample - k
    return output


def make_cat_map(labels, ca_coors, data_shape, downsample=1):
    output = np.zeros((1, *[int(np.ceil(i / downsample)) for i in data_shape]))
    for (label, ca_coor) in zip(labels, ca_coors):
        i = int(ca_coor[0] / downsample)
        j = int(ca_coor[1] / downsample)
        k = int(ca_coor[2] / downsample)
        output[0, i, j, k] = label
    return output


def transform_data(box, label_coor, label_rot):
    random_seqs = np.random.randint(3, size=3)
    box = rotate_data(box, 2, 3, random_seqs[0])
    box = rotate_data(box, 1, 3, random_seqs[1])
    box = rotate_data(box, 1, 2, random_seqs[2])
    label_coor = rotate_label(label_coor, 2, 3, random_seqs[0])
    label_coor = rotate_label(label_coor, 1, 3, random_seqs[1])
    label_coor = rotate_label(label_coor, 1, 2, random_seqs[2])
    label_rot = rotate_data(label_rot, 2, 3, random_seqs[0])
    label_rot = rotate_data(label_rot, 1, 3, random_seqs[1])
    label_rot = rotate_data(label_rot, 1, 2, random_seqs[2])
    return box, label_coor, label_rot


def rotate_data(arr, axis0, axis1, mode):
    if mode == 0:
        pass
    elif mode == 1:
        arr = torch.transpose(arr, axis0, axis1) #.clone()
        arr = torch.flip(arr, [axis0]) #.clone()
    elif mode == 2:
        arr = torch.transpose(arr, axis0, axis1) #.clone()
        arr = torch.flip(arr, [axis1]) #.clone()
    else:
        arr = torch.flip(arr, [axis0, axis1]) #.clone()
    return arr


def rotate_label(arr, axis0, axis1, mode):
    if mode == 0:
        pass
    elif mode == 1:
        arr = torch.transpose(arr, axis0, axis1) #.clone()
        arr = torch.flip(arr, [axis0]) #.clone()
        arr = swap_coors(arr, axis0, axis1) #.clone()
        arr = swap_coors(arr, axis0+3, axis1+3) #.clone()
        arr = swap_coors(arr, axis0+6, axis1+6) #.clone()
        arr = swap_coors(arr, axis0+9, axis1+9) #.clone()
        arr[axis0, :, :, :] = 1.0 - arr[axis0, :, :, :]
        arr[axis0+3, :, :, :] = 0.0 - arr[axis0+3, :, :, :]
        arr[axis0+6, :, :, :] = 0.0 - arr[axis0+6, :, :, :]
        arr[axis0+9, :, :, :] = 0.0 - arr[axis0+9, :, :, :]
    elif mode == 2:
        arr = torch.transpose(arr, axis0, axis1) #.clone()
        arr = torch.flip(arr, [axis1]) #.clone()
        arr = swap_coors(arr, axis0, axis1) #.clone()
        arr = swap_coors(arr, axis0+3, axis1+3) #.clone()
        arr = swap_coors(arr, axis0+6, axis1+6) #.clone()
        arr = swap_coors(arr, axis0+9, axis1+9) #.clone()
        arr[axis1, :, :, :] = 1.0 - arr[axis1, :, :, :]
        arr[axis1+3, :, :, :] = 0.0 - arr[axis1+3, :, :, :]
        arr[axis1+6, :, :, :] = 0.0 - arr[axis1+6, :, :, :]
        arr[axis1+9, :, :, :] = 0.0 - arr[axis1+9, :, :, :]
    else:
        arr = torch.flip(arr, [axis0, axis1]) #.clone()
        arr[axis0, :, :, :] = 1.0 - arr[axis0, :, :, :]
        arr[axis0+3, :, :, :] = 0.0 - arr[axis0+3, :, :, :]
        arr[axis0+6, :, :, :] = 0.0 - arr[axis0+6, :, :, :]
        arr[axis0+9, :, :, :] = 0.0 - arr[axis0+9, :, :, :]
        arr[axis1, :, :, :] = 1.0 - arr[axis1, :, :, :]
        arr[axis1+3, :, :, :] = 0.0 - arr[axis1+3, :, :, :]
        arr[axis1+6, :, :, :] = 0.0 - arr[axis1+6, :, :, :]
        arr[axis1+9, :, :, :] = 0.0 - arr[axis1+9, :, :, :]
    return arr


def swap_coors(arr, axis0, axis1):
    tmp = arr[axis0, :, :, :].clone()
    arr[axis0, :, :, :] = arr[axis1, :, :, :].clone()
    arr[axis1, :, :, :] = tmp.clone()
    return arr


def nms(ca_confs, ca_coors, min_dist_cutoff=3.8):
    idxs = torch.argsort(ca_confs, dim=0, descending=True)
    filtered_idxs = []
    filtered_idxs.append(idxs[0])
    for idx in idxs[1:]:
        dists = torch.norm(ca_coors[idx, :] - ca_coors[filtered_idxs, :], dim=1)
        min_dist = torch.min(dists)
        if min_dist < min_dist_cutoff:
            pass
        else:
            filtered_idxs.append(idx)
    return filtered_idxs

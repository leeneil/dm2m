import os
import glob
import time
import json
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from aa_code_utils import *
from dm_dataset import DeepMapDataset
from dm_net import DeepMapNet
from dm_loss import DeepMapLoss
from dm_utils import *


def train(model, ds, params, device):
    model = model.to(device)
    loss_fn = DeepMapLoss(
        ca_weight=params["loss"]["ca_weight"],
        c_weight=params["loss"]["c_weight"],
        n_weight=params["loss"]["n_weight"],
        o_weight=params["loss"]["o_weight"],
        cat_weight=params["loss"]["cat_weight"],
        pos_weight_factor=params["loss"]["pos_weight_factor"],
        c_ca=params["loss"]["c_ca"],
        n_ca=params["loss"]["n_ca"],
        o_ca=params["loss"]["o_ca"],
        rot_label=ds.rot_label,
        device=device
    )
    metric_fn = Metrics(
        c_ca=params["loss"]["c_ca"],
        n_ca=params["loss"]["n_ca"],
        o_ca=params["loss"]["o_ca"],
        cutoff=params["loss"]["cutoff"]
    )
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]
    shuffle = params["shuffle"]
    n_ep = params["n_epochs"]
    lr = params["lr"]
    backprop_every = params["backprop_every"]
    backup_every = params["backup_every"]
    backup_path = params["backup_path"]
    dl = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    seen = 0
    for i in range(n_ep):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for _j, (x, y_coor, y_rot) in enumerate(dl):
            if torch.sum(y_coor[:, 0, :, :, :]) < 2:
                continue
            x = x.to(device)
            y_coor = y_coor.to(device)
            y_rot = y_rot.to(device)
            model.train()
            output = model(x.float())
            seen += x.size(0)
            loss = loss_fn(
                outputs=output,
                ca_targets=y_coor[:, 0:4, :, :, :].float(),
                c_targets=y_coor[:, 4:7, :, :, :].float(),
                n_targets=y_coor[:, 7:10, :, :, :].float(),
                o_targets=y_coor[:, 10:13, :, :, :].float(),
                rot_targets=y_rot.long()
            )
            loss.backward()
            if seen % backprop_every == 0:
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                print("{:6d} loss {:8.4f}".format(seen, loss.item()))
                n_gt, n_p, pre, rec, ca_err, c_err, n_err, o_err, rot_cat = metric_fn(
                    output=output,
                    ca_targets=y_coor[:, 0:4, :, :, :].float(),
                    c_targets=y_coor[:, 4:7, :, :, :].float(),
                    n_targets=y_coor[:, 7:10, :, :, :].float(),
                    o_targets=y_coor[:, 10:13, :, :, :].float(),
                    rot_targets=y_rot.long()
                )
                print("       GT{:4d}  detected{:6d}  precision {:.3f}  recall {:.3f}  rot {:.3f}"
                      .format(n_gt, n_p, pre, rec, rot_cat))
                print("       CA {:.3f}  C {:.3f}  N {:.3f}  O {:.3f}".format(ca_err, c_err, n_err, o_err))
                if seen % backup_every == 0:
                    torch.save(model.state_dict(), os.path.join(backup_path, "latest.pt"))
        with torch.no_grad():
            torch.save(model.state_dict(), os.path.join(backup_path, "ep_{:03d}.pt".format(i)))


def val(model, ds, params, device):
    debug = False
    model = model.to(device)
    model.eval()
    # model.train()
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]
    metric_fn = Metrics(
        c_ca=params["loss"]["c_ca"],
        n_ca=params["loss"]["n_ca"],
        o_ca=params["loss"]["o_ca"],
        cutoff=params["loss"]["cutoff"]
    )
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    data = []
    with torch.no_grad():
        for _j, (x, y_coor, y_rot) in enumerate(dl):
            x = x.to(device)
            y_coor = y_coor.to(device)
            y_rot = y_rot.to(device)
            output = model(x.float())
            n_gt, n_p, pre, rec, ca_err, c_err, n_err, o_err, rot_cat = metric_fn(
                output=output,
                ca_targets=y_coor[:, 0:4, :, :, :].float(),
                c_targets=y_coor[:, 4:7, :, :, :].float(),
                n_targets=y_coor[:, 7:10, :, :, :].float(),
                o_targets=y_coor[:, 10:13, :, :, :].float(),
                rot_targets=y_rot.long()
            )
            print("       GT{:4d}  detected{:6d}  precision {:.3f}  recall {:.3f}  rot {:.3f}"
                  .format(n_gt, n_p, pre, rec, rot_cat))
            print("       CA {:.3f}  C {:.3f}  N {:.3f}  O {:.3f}".format(ca_err, c_err, n_err, o_err))
            if debug:
                output_np = output.data.cpu().numpy()
                input_np = x.data.cpu().numpy()
                label_np = y_coor.data.cpu().numpy()
                np.save(f"debug/val_x_{_j}.npy", input_np)
                np.save(f"debug/val_y_{_j}.npy", label_np)
                np.save(f"debug/val_z_{_j}.npy", output_np)
            data.append(
                dict(
                    n_gt=n_gt,
                    n_p=n_p,
                    precision=pre,
                    recall=rec,
                    ca_err=ca_err,
                    c_err=c_err,
                    n_err=n_err,
                    o_err=o_err,
                    rotamer_precision=rot_cat,
                    filename=f"debug/val_y_{_j}.npy"
                )
            )
    df = pd.DataFrame(data)
    print(df.describe())
    df.to_csv("debug/report.csv")


def predict(model, ds, params, device):
    model = model.to(device)
    model.eval()
    # model.train()
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    data = []
    anchors = (
        params["loss"]["c_ca"],
        params["loss"]["n_ca"],
        params["loss"]["o_ca"]
    )
    with torch.no_grad():
        for _j, (x, y_coor, y_rot) in enumerate(dl):
            x = x.to(device)
            y_coor = y_coor.to(device)
            y_rot = y_rot.to(device)
            output = model(x.float())
            ca_conf, ca_coors, c_coors, n_coors, o_coors, rot_cat = extract(
                output[0, :, :, :, :],
                anchors=anchors,
                cutoff=params["cutoff"]
            )
            if ca_conf.size(0) == 0:
                print("[{:3d}] raw{:6d} nms{:6d}".format(_j, 0, 0))
            else:
                nms_idxs = nms(
                    ca_conf,
                    ca_coors,
                    min_dist_cutoff=params["min_dist_cutoff"]
                )
                print(
                    "[{:3d}] raw{:6d} nms{:6d}".format(
                        _j,
                        ca_conf.size(0),
                        len(nms_idxs)
                    )
                )
                ca_conf = ca_conf[nms_idxs, :].data.cpu().numpy()
                ca_coors = ca_coors[nms_idxs, :].data.cpu().numpy()
                c_coors = c_coors[nms_idxs, :].data.cpu().numpy()
                n_coors = n_coors[nms_idxs, :].data.cpu().numpy()
                o_coors = o_coors[nms_idxs, :].data.cpu().numpy()
                rot_cat = rot_cat[nms_idxs, :].data.cpu().numpy()
                data = np.concatenate(
                    (ca_conf, ca_coors, c_coors, n_coors, o_coors, rot_cat),
                    axis=1
                )
                df = pd.DataFrame(data)
                df.to_csv("predict/{}.csv".format(str(_j).zfill(9)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", type=str, default=None, help="train | validate | predict")
    p.add_argument("params", type=str, default=None, help="JSON file with parameters")
    p.add_argument("--model", "-m", type=str, default=None, help="Path to a trained model")
    p.add_argument("--device", "-d", type=int, default=0, help="GPU device number")
    p.add_argument("--verbose", "-v", action="store_true", help="Be verbose")
    return p.parse_args()


def main():
    args = parse_args()
    params = json.load(open(args.params))
    seed = params["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = DeepMapNet(
        n_out=(33, 175)[params["rot_label"]],
        n_filters=params["net"]["n_filters"],
        bottleneck=params["net"]["bottleneck"],
        track_running_stats=params["net"]["track_running_stats"]
    )
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        print("Trained model {} loaded".format(args.model))
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device(f"cuda:{args.device}")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    if args.command == "train":
        print("Train DeepMap")
        pdb_list = params["train"]["pdb_list"]
        data_root = params["train"]["data_root"]
        ds_train = DeepMapDataset(
            pdb_list,
            data_root,
            rot_label=params["rot_label"],
            normalize=params["train"]["normalize"],
            transform=params["train"]["transform"]
        )
        print("Train set: {} boxes".format(len(ds_train)))
        train(model, ds_train, params["train"], device)
    elif args.command == "validate":
        print("Validate DeepMap")
        pdb_list = params["val"]["pdb_list"]
        data_root = params["val"]["data_root"]
        ds_val = DeepMapDataset(
            pdb_list,
            data_root,
            rot_label=params["rot_label"],
            normalize=True
        )
        print("Validation set: {} boxes".format(len(ds_val)))
        val(model, ds_val, params["val"], device)
    elif args.command == "predict":
        print("Predict with DeepMap")
        pdb_list = params["predict"]["pdb_list"]
        data_root = params["predict"]["data_root"]
        ds_pre = DeepMapDataset(
            pdb_list,
            data_root,
            rot_label=params["rot_label"],
            normalize=True,
            build_report=True
        )
        print("Prediction: {} boxes".format(len(ds_pre)))
        predict(model, ds_pre, params["predict"], device)


if __name__ == "__main__":
    main()

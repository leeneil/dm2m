import torch
import torch.nn as nn


def accuracy(scores, targets, ignore_index=162):
    val, ind = torch.max(scores, 1)
    acc = (ind[targets != ignore_index].squeeze().long() == targets[targets != ignore_index].squeeze().long()).sum()
    # print("n correct", acc.cpu().item())
    acc = acc.item() / scores.size(0)
    return acc


def ca_zyx_error(outputs, targets, mask):
    sig = nn.Sigmoid()
    z = sig(outputs[:, 0, :, :, :][mask]) - targets[:, 0, :, :, :][mask]
    y = sig(outputs[:, 1, :, :, :][mask]) - targets[:, 1, :, :, :][mask]
    x = sig(outputs[:, 2, :, :, :][mask]) - targets[:, 2, :, :, :][mask]
    err = float(torch.mean(torch.norm(torch.cat((z.view(-1, 1), y.view(-1, 1), x.view(-1, 1)), dim=1), dim=1)))
    return err


def zyx_error(outputs, targets, mask, anchor):
    # dist = anchor * torch.exp(torch.tanh(outputs[:, 0, :, :, :][mask]))
    z = torch.tanh(outputs[:, 0, :, :, :][mask]).view(-1, 1)
    y = torch.tanh(outputs[:, 1, :, :, :][mask]).view(-1, 1)
    x = torch.tanh(outputs[:, 2, :, :, :][mask]).view(-1, 1)
    # delta = torch.norm(torch.cat((z, y, x), dim=1), dim=1)
    z = anchor * z - targets[:, 0, :, :, :][mask].view(-1, 1)
    y = anchor * y - targets[:, 1, :, :, :][mask].view(-1, 1)
    x = anchor * x - targets[:, 2, :, :, :][mask].view(-1, 1)
    err = float(torch.mean(torch.norm(torch.cat((z, y, x), dim=1), dim=1)))
    return err


class AAAccuracy(object):

    def __init__(self, matrix_path, device=None):
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.mat = torch.from_numpy(np.load(matrix_path, allow_pickle=True)).float()
        self.mat = self.mat.to(self.device)

    def __call__(self, scores, targets, ignore_index=20):
        scores = torch.mm(scores.squeeze(), self.mat)
        scores, _ = torch.max(scores.view(-1, 21, 34), 2)
        _, idxs = torch.max(scores.view(-1, 21), 1)
        gt = targets  # .cpu().data.numpy()
        pr = idxs  # .cpu().data.numpy()
        acc = torch.sum(gt[gt != ignore_index].long() == pr[gt != ignore_index].long())
        # print("GT", targets.cpu().data.numpy())
        # print("PR", idxs.cpu().data.numpy())
        acc = acc.double() / scores.size(0)  # float(scores.size(0))
        return acc


class Metrics(object):

    def __init__(self, c_ca, n_ca, o_ca, cutoff=0.1, dx=0.25, downsample=8):
        self.anchors = [c_ca, n_ca, o_ca]
        self.cutoff = cutoff
        self.scale = downsample * dx

    def __call__(self, output, ca_targets, c_targets, n_targets, o_targets, rot_targets):
        sig = nn.Sigmoid()
        ca_conf = sig(output[:, 0, :, :, :])
        n_gt = int(torch.sum(ca_targets[:, 0, :, :, :]))
        positives = ca_conf > self.cutoff
        n_p = int(torch.sum(positives.float()))
        n_tp = torch.sum((ca_targets[:, 0, :, :, :])[positives])
        precision = float(n_tp / max(1, n_p))
        recall = float(n_tp / n_gt)
        mask = ca_targets[:, 0, :, :, :] > 0
        ca_err = self.scale * ca_zyx_error(output[:, 1:4, :, :, :], ca_targets[:, 1:4, :, :, :], mask)
        c_err = zyx_error(output[:, 4:7, :, :, :], c_targets, mask, self.anchors[0])
        n_err = zyx_error(output[:, 7:10, :, :, :], n_targets, mask, self.anchors[1])
        o_err = zyx_error(output[:, 10:13, :, :, :], o_targets, mask, self.anchors[2])
        rot_cat = 0
        if torch.sum(mask.float()) > 0:
            n_rot_labels = output.size(1) - 13
            mask_expand = mask[:, None, :, :, :].expand(
                mask.size(0),
                n_rot_labels,
                mask.size(1),
                mask.size(2),
                mask.size(3)
            )
            rot_cat = accuracy(
                torch.transpose(
                    output[:, 13:(13+n_rot_labels), :, :, :][mask_expand]\
                        .view(n_rot_labels, -1),
                    0,
                    1
                ),
                rot_targets[mask[:, None, :, :, :]].view(-1),
                ignore_index=n_rot_labels
            )
        return n_gt, n_p, precision, recall, ca_err, c_err, n_err, o_err, rot_cat


def extract_zyx_coors(outputs, mask, ca_coors, anchor):
    z = torch.tanh(outputs[0, :, :, :][mask]).view(-1, 1)
    y = torch.tanh(outputs[1, :, :, :][mask]).view(-1, 1)
    x = torch.tanh(outputs[2, :, :, :][mask]).view(-1, 1)
    coors = anchor * torch.cat((z, y, x), dim=1)
    coors = coors + ca_coors
    return coors


def extract(scores, anchors, cutoff=0.5, downsample=8, dx=0.25):
    ca_scores = nn.Sigmoid()(scores[0:4, :, :, :].data)
    mask = ca_scores[0, :, :, :] > cutoff
    # print("mask", mask.size())
    ca_conf = ca_scores[0, :, :, :][mask].view(-1, 1)
    # print("conf", conf.size())
    uvw = torch.nonzero(mask)
    # print("uwv", uvw.size())
    u = ca_scores[1, :, :, :][mask] + uvw[:, 0].float()
    v = ca_scores[2, :, :, :][mask] + uvw[:, 1].float()
    w = ca_scores[3, :, :, :][mask] + uvw[:, 2].float()
    ca_coors = downsample * dx * torch.cat(
        (u.view(-1, 1), v.view(-1, 1), w.view(-1, 1)),
        dim=1
    )
    c_coors = extract_zyx_coors(
        scores[4:7, :, :, :], mask, ca_coors, anchors[0]
    )
    n_coors = extract_zyx_coors(
        scores[7:10, :, :, :], mask, ca_coors, anchors[1]
    )
    o_coors = extract_zyx_coors(
        scores[10:13, :, :, :], mask, ca_coors, anchors[2]
    )
    # print("ca_coors", ca_coors.size())
    n_rot_labels = scores.size(0) - 13
    cat = scores[13:, :, :, :][mask[None, :, :, :]\
        .expand(n_rot_labels, mask.size(0), mask.size(1), mask.size(2))]
    cat = torch.transpose(cat.view(n_rot_labels, -1), 0, 1)
    # print("cat", cat.size())
    return ca_conf, ca_coors, c_coors, n_coors, o_coors, cat

import torch
import torch.nn as nn


class DeepMapLoss(nn.Module):

    def __init__(
        self, ca_weight=20, cat_weight=20, o_weight=10, c_weight=10,
        n_weight=10, pos_weight_factor=1.0, c_ca=1.56, n_ca=1.51, o_ca=2.43,
        rot_label=True, device="cpu"
    ):
        super(DeepMapLoss, self).__init__()
        self.bce = None
        self.mse = nn.MSELoss(reduction="mean")
        self.n_rot_labels = (20, 162)[rot_label]
        self.ce = nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=self.n_rot_labels
        )
        self.ca_weight = ca_weight
        self.c_weight = c_weight
        self.n_weight = n_weight
        self.o_weight = o_weight
        self.cat_weight = cat_weight
        self.pos_weight_factor = pos_weight_factor
        self.anchors = (c_ca, n_ca, o_ca)
        self.sig = nn.Sigmoid()
        self.rot_label = rot_label
        self.device = device

    def forward(
            self,
            outputs,
            o_targets,
            c_targets,
            ca_targets,
            n_targets,
            rot_targets
        ):
        pos_weight = self.pos_weight_factor   \
                     * torch.sum(ca_targets[:, 0, :, :, :] < 1).float()\
                     / torch.sum(ca_targets[:, 0, :, :, :] > 0).float()
        pos_weight = torch.clamp(pos_weight, min=1.0, max=10000.0)
        # print("pos weight", float(pos_weight.data))
        pos_weight = pos_weight.to(self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        ca_conf_loss = self.bce(outputs[:, 0, :, :, :], ca_targets[:, 0, :, :, :])
        loss = ca_conf_loss
        mask = ca_targets[:, 0, :, :, :] > 0
        if torch.sum(mask.float()) > 0:
            # CA
            ca_z_loss = self.mse(
                self.sig(outputs[:, 1, :, :, :][mask]),
                ca_targets[:, 1, :, :, :][mask]
            )
            ca_y_loss = self.mse(
                self.sig(outputs[:, 2, :, :, :][mask]),
                ca_targets[:, 2, :, :, :][mask]
            )
            ca_x_loss = self.mse(
                self.sig(outputs[:, 3, :, :, :][mask]),
                ca_targets[:, 3, :, :, :][mask]
            )
            # C
            c_z = self.anchors[0] * torch.tanh(outputs[:, 4, :, :, :][mask])
            c_y = self.anchors[0] * torch.tanh(outputs[:, 5, :, :, :][mask])
            c_x = self.anchors[0] * torch.tanh(outputs[:, 6, :, :, :][mask])
            c_z_loss = self.mse(c_z, (c_targets[:, 0, :, :, :][mask]))
            c_y_loss = self.mse(c_y, (c_targets[:, 1, :, :, :][mask]))
            c_x_loss = self.mse(c_x, (c_targets[:, 2, :, :, :][mask]))
            # N
            n_z = self.anchors[1] * torch.tanh(outputs[:, 7, :, :, :][mask])
            n_y = self.anchors[1] * torch.tanh(outputs[:, 8, :, :, :][mask])
            n_x = self.anchors[1] * torch.tanh(outputs[:, 9, :, :, :][mask])
            n_z_loss = self.mse(n_z, (n_targets[:, 0, :, :, :][mask]))
            n_y_loss = self.mse(n_y, (n_targets[:, 1, :, :, :][mask]))
            n_x_loss = self.mse(n_x, (n_targets[:, 2, :, :, :][mask]))
            # O
            o_z = self.anchors[2] * torch.tanh(outputs[:, 10, :, :, :][mask])
            o_y = self.anchors[2] * torch.tanh(outputs[:, 11, :, :, :][mask])
            o_x = self.anchors[2] * torch.tanh(outputs[:, 12, :, :, :][mask])
            o_z_loss = self.mse(o_z, (o_targets[:, 0, :, :, :][mask]))
            o_y_loss = self.mse(o_y, (o_targets[:, 1, :, :, :][mask]))
            o_x_loss = self.mse(o_x, (o_targets[:, 2, :, :, :][mask]))
            if self.rot_label:
                n_rot_labels = 162
            else:
                n_rot_labels = 20
            mask_expand = mask[:, None, :, :, :].expand(
                mask.size(0),
                self.n_rot_labels,
                mask.size(1),
                mask.size(2),
                mask.size(3),
            )
            cat_loss = self.ce(
                torch.transpose(
                    outputs[:, 13:(13+self.n_rot_labels), :, :, :][mask_expand]\
                    .view(self.n_rot_labels, -1),
                    0,
                    1,
                ),
                rot_targets[mask[:, None, :, :, :]].view(-1)
            )
            ca_zyx_loss = ca_z_loss + ca_y_loss + ca_x_loss
            c_zyx_loss = c_z_loss + c_y_loss + c_x_loss
            n_zyx_loss = n_z_loss + n_y_loss + n_x_loss
            o_zyx_loss = o_z_loss + o_y_loss + o_x_loss
            loss += (
                self.ca_weight * ca_zyx_loss +
                self.c_weight * c_zyx_loss +
                self.n_weight * n_zyx_loss +
                self.o_weight * o_zyx_loss +
                self.cat_weight * cat_loss
            )
        return loss

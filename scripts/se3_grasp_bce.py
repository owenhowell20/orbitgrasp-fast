import torch
import numpy as np
from models.orbitgrasp_unet import EquiformerUnet


class OrbitGrasp:
    def __init__(self, device, num_channel=36, lr=1e-4, lmax=3, mmax=2):
        self.device = device
        self.net = EquiformerUnet(lmax_list=[lmax], mmax_list=[mmax]).to(device)
        self.parameters = list(self.net.parameters())
        self.optim = torch.optim.AdamW(
            [{"params": self.net.parameters(), "lr": lr}], weight_decay=3e-4
        )
        self.iters = 0
        self.lmax = lmax
        self.mmax = mmax
        self.num_channel = num_channel

        print(
            "mask grasp params: ",
            sum(p.numel() for p in self.parameters if p.requires_grad),
        )

    def forward(self, feature_points, grasp_indices, spherical_harmonics, train=True):

        if train:
            self.net.train()
        else:
            self.net.eval()

        with torch.set_grad_enabled(train):
            features = self.net(feature_points)

        grasp_coefficients = features.embedding[grasp_indices].squeeze(-1)
        spherical_harmonics = spherical_harmonics.squeeze(0).to(
            self.device, dtype=torch.float
        )

        results = torch.bmm(
            grasp_coefficients.unsqueeze(1), spherical_harmonics.transpose(1, 2)
        )
        results = results.squeeze(1).contiguous()
        # results = torch.einsum('nk,nmk->nm', grasp_coefficients, spherical_harmonics).contiguous()
        return results

    def train(self, batch, backprop=True, balance=True):
        feature_pcd, grasp_indices, labels, harmonics = batch  # N; [M,P], [M,P,I]
        labels = labels.view(-1)

        results = self.forward(feature_pcd, grasp_indices, harmonics, train=True).view(
            -1
        )

        if balance:

            positive_indices = (labels == 1).nonzero().squeeze(1)
            negative_indices = (labels == 0).nonzero().squeeze(1)

            if positive_indices.numel() == 0 or negative_indices.numel() == 0:
                return None

            smaller_group = min(positive_indices.size(0), negative_indices.size(0))
            positive_indices = positive_indices[
                torch.randperm(positive_indices.size(0))[:smaller_group]
            ]
            negative_indices = negative_indices[
                torch.randperm(negative_indices.size(0))[:smaller_group]
            ]
            balanced_index = torch.cat([positive_indices, negative_indices])

            results = results[balanced_index]
            labels = labels[balanced_index]

        if results.numel() > 0:  # Ensure there are elements to compute the loss on
            loss_bce = torch.nn.BCEWithLogitsLoss()
            loss = loss_bce(results, labels)

            # Backpropagation
            if backprop:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            self.iters += 1

            return np.float32(loss.item())
        else:
            return None

    def test(self, batch):
        feature_pcd, grasp_indices, labels, harmonics = batch  # N; [M,P], [M,P,I]

        grasp_indices = torch.cat(grasp_indices, dim=0)
        labels = torch.cat(labels, dim=1).squeeze(0)
        loss_bce = torch.nn.BCEWithLogitsLoss()
        # avg_loss = None
        # total_loss = torch.tensor(0., requires_grad=True, device=self.device)

        labels = labels.view(-1)

        negative_indices = torch.where(labels == 0)[0]
        positive_indices = torch.where(labels == 1)[0]

        positive_mask = (labels == 1).sum().item()
        negative_mask = (labels == 0).sum().item()

        if positive_mask == 0 or negative_mask == 0:
            return None, None, None
        results = self.forward(feature_pcd, grasp_indices, harmonics, train=False)
        results = results.view(-1)
        test_index = torch.cat((positive_indices, negative_indices), dim=0)
        results = results[test_index]
        labels = labels[test_index]

        if results.shape[0] == 0:
            return None, None, None
        loss = loss_bce(results, labels)

        preds = (results > 0).float()
        correct = (preds == labels).float().sum()
        accuracy = correct / len(labels)

        max_index = torch.argmax(results)
        max_pred_correct = (preds[max_index] == labels[max_index]).float()

        return (
            np.float32(loss.item()),
            np.float32(accuracy.item()),
            np.float32(max_pred_correct.item()),
        )

    def predict(self, feature_pcds, grasp_indices, spherical_harmonics):
        self.net.eval()
        with torch.no_grad():
            features = self.net(feature_pcds)
            results_list = []
            features_list = []

            sizes = [feature_pcd.x.shape[0] for feature_pcd in feature_pcds]
            offsets = torch.cumsum(torch.tensor([0] + sizes[:-1]), dim=0).to(
                self.device
            )

            for i, feature_pcd in enumerate(feature_pcds):
                cur_features = features.embedding[offsets[i] : offsets[i] + sizes[i]]
                sh = spherical_harmonics[i].detach().float().to(self.device)
                grasp_coefficients = cur_features[grasp_indices[i]].squeeze(-1)
                results = torch.einsum(
                    "nk,nmk->nm", grasp_coefficients, sh
                ).contiguous()
                results_list.append(results)
                features_list.append(grasp_coefficients)
            return results_list, features_list

    def save(self, filename):
        self.net.eval()
        torch.save(self.net.state_dict(), filename)

    def load(self, path):
        self.net.eval()
        self.net.load_state_dict(torch.load(path, self.device))

    def loadFromState(self, state):
        self.net.eval()
        self.net.load_state_dict(state)

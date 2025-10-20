# pointnet_rgb.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STN3d(nn.Module):
    """ Spatial Transformer Network for 3D or multi-channel input """
    def __init__(self, input_dim=3):
        super(STN3d, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim*input_dim)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batchsize, -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.input_dim, dtype=torch.float32, device=x.device).view(1,self.input_dim*self.input_dim).repeat(batchsize,1)
        x = x + iden
        x = x.view(-1, self.input_dim, self.input_dim)
        return x

class PointNetFeat(nn.Module):
    """ PointNet feature extractor for embedding """
    def __init__(self, input_dim=6, global_feat=True, feature_transform=False, emb_dim=512):
        super(PointNetFeat, self).__init__()
        self.stn = STN3d(input_dim=input_dim)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, emb_dim, 1)  # embedding dimension
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(emb_dim)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        # if self.feature_transform:
        #     self.fstn = STNkd(k=64)

    def forward(self, x):
        # x: [B, C, N]
        n_pts = x.size(2)
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=False)[0]  # global max pooling
        if self.global_feat:
            return x  # [B, emb_dim]
        else:
            x = x.unsqueeze(2).repeat(1,1,n_pts)
            return torch.cat([x, pointfeat],1)  # [B, emb_dim+64, N]

class PointNetEmbedding(nn.Module):
    """ PointNet returns embedding for contrastive learning """
    def __init__(self, input_dim=6, emb_dim=512, feature_transform=False):
        super(PointNetEmbedding, self).__init__()
        self.feat = PointNetFeat(input_dim=input_dim, global_feat=True, feature_transform=feature_transform, emb_dim=emb_dim)

    def forward(self, x):
        # x: [B, C, N]
        emb = self.feat(x)
        return emb  # [B, emb_dim]

if __name__ == "__main__":
    sim_data = torch.rand(4,6,2048)  # batch 4, 6 channels, 2048 points
    model = PointNetEmbedding(input_dim=6, emb_dim=512)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    out = model(sim_data)
    print("Output embedding shape:", out.shape)  # [4, 512]

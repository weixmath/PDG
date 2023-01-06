import torch
import torch.nn as nn
class Feature_Extractor(nn.Module):
    def __init__(self, emb_dims):
        super(Feature_Extractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.ReLU(),
        )
    def forward(self, xyz, pn=None):
        batch_size = xyz.shape[0]
        point_num = xyz.shape[2]
        feat = self.encoder(xyz)
        if pn is not None:
            index = torch.arange(point_num).reshape(1, -1).repeat(batch_size, 1).to('cuda')
            mask = (index < pn.reshape(-1, 1)).unsqueeze(1)
            feat = feat.masked_fill(torch.bitwise_not(mask).repeat(1, 1024, 1), 0)
        return feat


class Classifier(nn.Module):
    def __init__(self, emb_dims, output_channels):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dims, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_channels)
        )

    def forward(self, x):
        return self.classifier(x)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=11):
        super(PointNet, self).__init__()
        self.args = args
        self.encoder = Feature_Extractor(args.emb_dims)
        self.cls = Classifier(args.emb_dims, output_channels)
    def forward(self, xyz, x_point_num=None):
        feat = self.encoder(xyz,x_point_num)
        ff1 = feat.max(2)[0]
        out = self.cls(ff1)
        return out
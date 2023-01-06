import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.model_utils import farthest_point_sample,index_points,knn2
from model.dgcnn import Feature_Extractor_DGCNN, Classifier_DGCNN
from model.pointnet import Feature_Extractor, Classifier

def soft_pool(x,cls):
    B,N,D = x.shape
    score_x = cls(x.reshape(-1,D))
    score_x_max = score_x.max(-1)[0].reshape(B,N,1)
    score_x_max = torch.sigmoid(score_x_max)
    pooled_x = torch.max(x*score_x_max,1)[0]
    return pooled_x,score_x


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        # q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        # k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        # v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        # out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class get_part_feat(nn.Module):
    def __init__(self, k=8, num_points=512):
        super(get_part_feat, self).__init__()
        self.CrossAtt = ScaledDotProductAttention(d_model=1024, d_k=1024, d_v=1024, h=1)
        self.k = k
        self.num_points = num_points
    def forward(self, xyz, feat_xyz, temp, cent_xyz=None):
        xyz = xyz.transpose(1, 2)
        feat_xyz = feat_xyz.transpose(1, 2)
        if cent_xyz == None:
            cent_xyz = index_points(xyz, farthest_point_sample(xyz, self.k))
        id = knn2(xyz, cent_xyz, self.num_points)
        # part_xyz = index_points(xyz, id)
        part_feat = index_points(feat_xyz, id)
        part_feat_max = torch.max(part_feat, 1)[0]
        transformed_part_feat = self.CrossAtt(part_feat_max, temp, temp)
        return transformed_part_feat, cent_xyz

class best(nn.Module):
    def __init__(self, args, output_channels=11):
        super(best, self).__init__()
        self.args = args
        model_name = args.model
        if model_name == 'pointnet':
            self.encoder = Feature_Extractor(args.emb_dims)
            self.cls = Classifier(args.emb_dims, output_channels)
            self.cls_part = Classifier(args.emb_dims, output_channels)
        elif model_name == 'dgcnn':
            self.encoder = Feature_Extractor_DGCNN(args.emb_dims)
            self.cls = Classifier_DGCNN(args.emb_dims, output_channels)
            self.cls_part = Classifier_DGCNN(args.emb_dims, output_channels)
        self.part_temp = nn.Parameter(torch.Tensor(384, 1024), requires_grad=True)
        torch.nn.init.xavier_normal_(self.part_temp)
        self.get_part_feat = get_part_feat(k=8, num_points=512)
        self.projection = nn.Sequential(
            nn.Linear(1024,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096,1024),
            nn.ReLU(),
        )
        # self.projection_all = nn.Sequential(OrderedDict([
        #     ('linear1', nn.Linear(1024,4096)),
        #     ('bn1',nn.BatchNorm1d(4096)),
        #     ('rl1',nn.ReLU()),
        #     ('linear2',nn.Linear(4096,1024)),
        #     ('rl2',nn.ReLU())
        # ]))
    def forward(self, xyz_q, pn_q=None, xyz_k=None, pn_k=None, m=0.0, test=True):
        B = xyz_q.shape[0]
        part_temp = (self.part_temp).unsqueeze(0).repeat(B, 1, 1)
        q1, cent_q1 = self.get_part_feat(xyz_q, self.encoder(xyz_q, pn_q), part_temp)
        feat_q1, score_q1 = soft_pool(q1,self.cls_part)
        out_q1 = self.cls(feat_q1)
        if test == True:
            return out_q1,score_q1
        else:
            q2,_ = self.get_part_feat(xyz_k, self.encoder(xyz_k, pn_k), part_temp,cent_q1)
            feat_q2, score_q2 = soft_pool(q2, self.cls_part)
            out_q2 = self.cls(feat_q2)
            query = F.normalize(self.projection(q1.reshape(-1,1024)),p=2,dim=-1)
            key = F.normalize(self.projection(q2.reshape(-1, 1024)), p=2, dim=-1)
            query_all = F.normalize(feat_q1.reshape(-1, 1024), p=2, dim=-1)
            key_all = F.normalize(feat_q2.reshape(-1, 1024), p=2, dim=-1)
            return out_q1, out_q2, score_q1,score_q2,torch.cat((query.unsqueeze(1),key.unsqueeze(1)),1),torch.cat((query_all.unsqueeze(1),key_all.unsqueeze(1)),1)




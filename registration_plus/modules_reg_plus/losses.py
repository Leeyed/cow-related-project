'''
@Description: loss functions
@Author: ronghuaiyang
@Date: 2019-05-24 11:06:27
@LastEditTime: 2019-05-24 11:47:37
@LastEditors: Please set LastEditors
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class MPSLoss(nn.Module):

    def __init__(self, m=0.5):
        super(MPSLoss, self).__init__()
        self.m = m

    def forward(self, input, target):
        fe_normed = F.normalize(input)
        fe_reshape = fe_normed.view(2, -1, fe_normed.size(1))
        fe_id = fe_reshape[0, :, :]
        fd_self = fe_reshape[1, :, :]
        cos_theta = torch.mm(fe_id, fd_self.transpose(0, 1))
        num_pairs = cos_theta.size(0)

        pos_mask = torch.eye(num_pairs, device=input.get_device())
        neg_mask = 1 - pos_mask
        pos_batch = torch.masked_select(cos_theta, pos_mask.byte())
        neg_batch1 = torch.masked_select(cos_theta, neg_mask.byte())
        neg_batch2 = torch.masked_select(cos_theta.transpose(0, 1), neg_mask.byte())

        neg_batch_1 = neg_batch1.view(num_pairs, -1)
        neg_batch_2 = neg_batch2.view(num_pairs, -1)
        neg_batch_1 = torch.max(neg_batch_1, 1)[0]
        neg_batch_2 = torch.max(neg_batch_2, 1)[0]

        neg_batch_max = torch.max(neg_batch_1, neg_batch_2)
        losses = F.relu(self.m + neg_batch_max - pos_batch)
        loss = torch.mean(losses)
        return loss


class MHELoss(nn.Module):
    def __init__(self, weight):
        super(MHELoss, self).__init__()
        self.weight = weight

    def forward(self, labels):
        w_norm = F.normalize(self.weight)
        mhe_loss = 0
        for i in labels:
            w_diff = w_norm[i] - w_norm
            loss = (w_diff * w_diff).sum()
            mhe_loss += loss
        mhe_loss = 1.0 / mhe_loss
        mhe_loss /= labels.size(0)
        mhe_loss /= (w_norm.size(0) - 1)
        return mhe_loss

def select_triplets_batch_hard(embeds, labels):
    dist_mtx = torch.cdist(embeds, embeds).detach().cpu().numpy()
    labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
    num = labels.shape[0]
    dia_inds = np.diag_indices(num)
    lb_eqs = labels == labels.T
    lb_eqs[dia_inds] = False
    dist_same = dist_mtx.copy()
    dist_same[lb_eqs == False] = -np.inf
    pos_idxs = np.argmax(dist_same, axis = 1)
    dist_diff = dist_mtx.copy()
    lb_eqs[dia_inds] = True
    dist_diff[lb_eqs == True] = np.inf
    neg_idxs = np.argmin(dist_diff, axis = 1)
    pos = embeds[pos_idxs].contiguous().view(num, -1)
    neg = embeds[neg_idxs].contiguous().view(num, -1)

    return embeds, pos, neg

def calc_cdist(a, b, metric='euclidean'):
    diff = a - b
    if metric == 'euclidean':
        return torch.sqrt(torch.sum(diff*diff, dim=1) + 1e-12)
    elif metric == 'sqeuclidean':
        return torch.sum(diff*diff, dim=1)
    elif metric == 'cityblock':
        return torch.sum(diff.abs(), dim=1)
    else:
        raise NotImplementedError("Metric %s has not been implemented!" % metric)

class TripletLoss(nn.Module):
    '''
    method: 'fix_margin', 'soft_margin'
    '''
    def __init__(self, margin=0.5, metric='sqeuclidean'):
        super(TripletLoss, self).__init__()

        self.margin = margin
        self.metric = metric


    # def triplet_margin_loss_root(self, anchor, positive, negative, p, margin):
    #     F.triplet_margin_loss
    # @staticmethod
    # def calc_cdist(a, b, metric='euclidean'):
    #     diff = a - b
    #     if metric == 'euclidean':
    #         return torch.sqrt(torch.sum(diff*diff, dim=1) + 1e-12)
    #     elif metric == 'sqeuclidean':
    #         return torch.sum(diff*diff, dim=1)
    #     elif metric == 'cityblock':
    #         return torch.sum(diff.abs(), dim=1)
    #     else:
    #         raise NotImplementedError("Metric %s has not been implemented!" % metric)

    # @staticmethod
    # def select_triplets_batch_hard(embeds, labels):
    #     dist_mtx = torch.cdist(embeds, embeds).detach().cpu().numpy()
    #     labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
    #     num = labels.shape[0]
    #     dia_inds = np.diag_indices(num)
    #     lb_eqs = labels == labels.T
    #     lb_eqs[dia_inds] = False
    #     dist_same = dist_mtx.copy()
    #     dist_same[lb_eqs == False] = -np.inf
    #     pos_idxs = np.argmax(dist_same, axis = 1)
    #     dist_diff = dist_mtx.copy()
    #     lb_eqs[dia_inds] = True
    #     dist_diff[lb_eqs == True] = np.inf
    #     neg_idxs = np.argmin(dist_diff, axis = 1)
    #     pos = embeds[pos_idxs].contiguous().view(num, -1)
    #     neg = embeds[neg_idxs].contiguous().view(num, -1)

    #     return embeds, pos, neg


    def forward(self, embeds, labels):
        anchors, positives, negatives = select_triplets_batch_hard(embeds, labels)
        ap_dist = calc_cdist(anchors, positives, metric=self.metric)
        an_dist = calc_cdist(anchors, negatives, metric=self.metric)

        if isinstance(self.margin, float):
            loss = F.relu(ap_dist - an_dist + self.margin).mean()
        elif self.margin.lower() == "soft":
            loss = F.softplus(ap_dist - an_dist).mean()
        else:
            raise NotImplementedError("The margin %s is not implemented in BatchHard!" % self.margin)        

        return loss


def convert_label_to_similarity(normed_feature, label):
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    # print(111,similarity_matrix.shape )
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
    # print(label.unsqueeze(1).shape)
    # print(label.unsqueeze(0).shape)
    # exit()

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    # print('similarity_matrix', similarity_matrix[:100])
    positive_matrix = positive_matrix.view(-1)
    # print('positive_matrix', similarity_matrix[positive_matrix].shape)
    negative_matrix = negative_matrix.view(-1)
    # print('similarity_matrix[negative_matrix]', similarity_matrix[negative_matrix].shape)
    # print('similarity_matrix[negative_matrix]', similarity_matrix[negative_matrix][:100])
    # print('similarity_matrix[positive_matrix]', similarity_matrix[positive_matrix][:100])
    # exit()
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]
    

###
# https://www.freesion.com/article/17321187110/
class CircleLoss(nn.Module):
    def __init__(self, m=0.4, gamma=80):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, embeds, labels):
        # max类内间距，min类间距离， sin： range(0,1)
        sp, sn = convert_label_to_similarity(embeds, labels)
        # values, indices = pred.topk(1, dim=1, largest=True, sorted=True)
        # print('sp min top 10:',sp.topk(10, dim=1, largest=True, sorted=True))
        # values, indices = torch.topk(-sp, 10)
        # print('sp min top 10:',-values)

        # values, indices = torch.topk(sn, 20)
        # print('sn max top 20:', values)

        # 将输入input张量每个元素的夹紧到区间 [min,max]
        # detach 官方解释是返回一个新的Tensor,从当前的计算图中分离出来
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        # print('logit_p', len(logit_p), logit_p.shape, logit_p)
        logit_n = an * (sn - delta_n) * self.gamma
        # print('logit_n', len(logit_n), logit_n.shape, logit_n.data)
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss




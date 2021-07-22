from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class DWI(nn.Module):
    def __init__(self, weight, alpha=1.0):
        super(DWI, self).__init__()
        self.alpha = 1.0
        self.weight = weight

    def forward(self, features, labels):
        with torch.no_grad():
            id_num = int(features.size(0) / 2)
            features_norm = F.normalize(features)
            features_norm_id = features_norm[: id_num]
            features_norm_self = features_norm[id_num:]
            weights_update = (features_norm_id + features_norm_self) / 2
            self.weight[labels[: id_num]] = F.normalize(weights_update)


class SVArcProduct(nn.Module):

    def __init__(self, feature_dim, class_number, s=30.0, m=0.5, t=1.2):
        super(SVArcProduct, self).__init__()
        self.s = s
        self.m = m
        self.t = t
        self.weight = Parameter(torch.FloatTensor(class_number, feature_dim))
        nn.init.xavier_uniform_(self.weight)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        theta = torch.acos(cos_theta)
        phi = torch.cos(theta + self.m)
        phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        h_t_theta = self.t * cos_theta + self.t - 1

        one_hot = cos_theta.new_zeros(cos_theta.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        f = torch.masked_select(phi, one_hot.byte()).view(-1, 1)
        h_t_theta_i = torch.where(f < cos_theta, h_t_theta, cos_theta)

        output = torch.where(one_hot > 0, phi, h_t_theta_i)
        output *= self.s

        return output


class SVAMProduct(nn.Module):
    def __init__(self, feature_dim, class_number, s=30.0, m=0.40, t=1.2):
        super(SVAMProduct, self).__init__()
        self.s = s
        self.m = m
        self.t = t
        # 在看过很多博客的时候发现了一个用法self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size)),
        # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        # 所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        # ————————————————
        # 版权声明：本文为CSDN博主「Danny明泽」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
        # 原文链接：https://blog.csdn.net/qq_36955294/article/details/88117170
        self.weight = Parameter(torch.FloatTensor(class_number, feature_dim))
        nn.init.xavier_uniform_(self.weight)
        # 357*512
        # print(self.weight.shape)
        # print(self.weight)
        # exit()

    # input: 64*512, 64
    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cos_theta - self.m
        h_t_theta = self.t * cos_theta + self.t - 1

        one_hot = cos_theta.new_zeros(cos_theta.size())
        # print('one_hot1')
        # for item in one_hot:
        #     print(item)
        # print('label.view(-1, 1).long()', label.view(-1, 1).long())

        # print('one_hot2')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # for item in one_hot:
        #     print(item)
        # print('one_hot', one_hot)

        f = torch.masked_select(phi, one_hot.bool()).view(-1, 1)
        # print('f', f)
        h_t_theta_i = torch.where(f < cos_theta, h_t_theta, cos_theta)
        # print('h_t_theta_i', h_t_theta_i)

        output = torch.where(one_hot > 0, phi, h_t_theta_i)
        # print('output', output)
        output *= self.s
        # print('output', output)
        # exit()
        return output


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, feature_dim, class_number, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(class_number, feature_dim))
        nn.init.xavier_uniform_(self.weight)

        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        theta = torch.acos(cos_theta)
        phi = torch.cos(theta + self.m)
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        one_hot = cos_theta.new_zeros(cos_theta.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = torch.where(one_hot > 0, phi, cos_theta)
        output *= self.s
        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, feature_dim, class_number, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.feature_dim = feature_dim
        self.class_number = class_number
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(class_number, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cos_theta - self.m
        one_hot = cos_theta.new_zeros(cos_theta.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = torch.where(one_hot > 0, phi, cos_theta)
        output *= self.s
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'feature_dim=' + str(self.feature_dim) \
               + ', class_number=' + str(self.class_number) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'


class AdaCos(nn.Module):

    def __init__(self, feature_dim, class_number):
        super(AdaCos, self).__init__()
        self.feature_dim = feature_dim
        self.class_number = class_number
        self.weight = Parameter(torch.FloatTensor(class_number, feature_dim))
        nn.init.xavier_uniform_(self.weight)

        self.sf = math.sqrt(2) * math.log(class_number - 1)
        self.sdt = self.sf
        self.th = math.cos(math.pi/4)

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))

        with torch.no_grad():
            one_hot = cos_theta.new_zeros(cos_theta.size())
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            N = label.size(0)
            b_avg = torch.where(one_hot > 0, torch.exp(self.sdt * cos_theta), torch.zeros_like(cos_theta))
            b_avg = b_avg.sum() / N
            cos_theta_label = torch.masked_select(cos_theta, one_hot.byte()).view(-1, 1)
            cos_theta_med = torch.median(cos_theta_label)
            cos_theta_0 = cos_theta_med if cos_theta_med > self.th else self.th
            self.sdt = torch.log(b_avg) / cos_theta_0
        output = self.sdt * cos_theta

        return output


# class AdaAm(nn.Module):

#     def __init__(self, feature_dim, class_number, m=0.4):
#         super(AdaAm, self).__init__()
#         self.feature_dim = feature_dim
#         self.class_number = class_number
#         self.weight = Parameter(torch.FloatTensor(class_number, feature_dim))
#         nn.init.xavier_uniform_(self.weight)

#         self.m = m
#         self.sf = math.sqrt(2) * math.log(class_number - 1)
#         self.sdt = self.sf
#         self.th = math.cos(math.pi/4)

#     def forward(self, input, label):
#         cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
#         one_hot = cos_theta.new_zeros(cos_theta.size())
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         phi = cos_theta - self.m

#         with torch.no_grad():
#             N = label.size(0)
#             b_avg = torch.where(one_hot > 0, torch.exp(self.sdt * phi), torch.zeros_like(phi))
#             b_avg = b_avg.sum() / N
#             cos_theta_label = torch.masked_select(phi, one_hot.byte()).view(-1, 1)
#             cos_theta_med = torch.median(cos_theta_label)
#             cos_theta_0 = cos_theta_med if cos_theta_med > self.th else self.th
#             self.sdt = torch.log(b_avg) / cos_theta_0

#         output = torch.where(one_hot > 0, phi, cos_theta)
#         output = self.sdt * output

#         return output


# class AdaSVAM(nn.Module):

#     def __init__(self, feature_dim, class_number,  m=0.4, t=1.2):
#         super(AdaSVAM, self).__init__()
#         self.feature_dim = feature_dim
#         self.class_number = class_number
#         self.weight = Parameter(torch.FloatTensor(class_number, feature_dim))
#         nn.init.xavier_uniform_(self.weight)

#         self.t = t
#         self.m = m
#         self.sdt = math.sqrt(2) * math.log(class_number - 1)
#         self.th = math.cos(math.pi/4)

#     def forward(self, input, label):
#         cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
#         one_hot = cos_theta.new_zeros(cos_theta.size())
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         phi = cos_theta - self.m

#         with torch.no_grad():
#             N = label.size(0)
#             b_avg = torch.where(one_hot > 0, torch.exp(self.sdt * phi), torch.zeros_like(phi))
#             b_avg = b_avg.sum() / N
#             cos_theta_label = torch.masked_select(phi, one_hot.byte()).view(-1, 1)
#             cos_theta_med = torch.median(cos_theta_label)
#             cos_theta_0 = cos_theta_med if cos_theta_med > self.th else self.th
#             self.sdt = torch.log(b_avg) / cos_theta_0

#         h_t_theta = self.t * cos_theta + self.t - 1
#         f = torch.masked_select(phi, one_hot.byte()).view(-1, 1)
#         h_t_theta_i = torch.where(f < cos_theta, h_t_theta, cos_theta)
#         output = torch.where(one_hot > 0, phi, h_t_theta_i)
#         output *= self.sdt

#         return output

    
# class P2SGrad(nn.Module):

#     def __init__(self, feature_dim, class_number):
#         super(P2SGrad, self).__init__()
#         self.feature_dim = feature_dim
#         self.class_number = class_number
#         self.weight = Parameter(torch.FloatTensor(class_number, feature_dim))
#         nn.init.xavier_uniform_(self.weight)
#         self.weight.backward(torch.zeros(self.weight.size()))

#     def forward(self, input, label):
#         with torch.no_grad():
#             eps = 1e-12
#             w_norm = self.weight.norm(p=2, dim=1, keepdim=True)
#             w_hat = self.weight / w_norm.clamp_min(eps).expand_as(self.weight)

#             x_norm = input.norm(p=2, dim=1, keepdim=True)
#             x_hat = input / x_norm.clamp_min(eps).expand_as(input)

#             cos_theta = F.linear(x_hat, w_hat)

#             one_hot = cos_theta.new_zeros(cos_theta.size())
#             one_hot.scatter_(1, label.view(-1, 1).long(), 1)

#             # shape: [batch class_num feature_dim]
#             x_norm = x_norm.unsqueeze(dim=2)
#             x_hat = x_hat.unsqueeze(dim=1)
#             w_norm = w_norm.unsqueeze(dim=0)
#             cos_theta = cos_theta.unsqueeze(dim=2)
#             one_hot = one_hot.unsqueeze(dim=2)

#             # print('x_norm', x_norm)

#             d_x = (w_hat - cos_theta * x_hat) / x_norm
#             d_w = (x_hat - cos_theta * w_hat) / w_norm

#             # print('d_x', d_x[0, label[0], :10])

#             grad_x = ((cos_theta - one_hot) * d_x).sum(dim=1)
#             grad_w = ((cos_theta - one_hot) * d_w).sum(dim=0)
#             self.weight._grad = grad_w

#             # if grad_x.norm().mean() < 1e-5:
#             #     grad_x *= 10

#             cos_theta_y = torch.masked_select(cos_theta, one_hot.byte())
#             theta_y = torch.acos(cos_theta_y).mean() / math.pi
#         return grad_x, theta_y


# def cal_grad(weight, x):
#     with torch.no_grad():
#         eps = 1e-12
#         w_norm = weight.norm(p=2, dim=1, keepdim=True)
#         w_hat = weight / w_norm.clamp_min(eps).expand_as(weight)

#         x_norm = input.norm(p=2, dim=1, keepdim=True)
#         x_hat = input / x_norm.clamp_min(eps).expand_as(x)

#         cos_theta = F.linear(x_hat, w_hat)

#         one_hot = cos_theta.new_zeros(cos_theta.size())
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)

#         # shape: [batch class_num feature_dim]
#         x_norm = x_norm.unsqueeze(dim=2)
#         x_hat = x_hat.unsqueeze(dim=1)
#         w_norm = w_norm.unsqueeze(dim=0)
#         cos_theta = cos_theta.unsqueeze(dim=2)
#         one_hot = one_hot.unsqueeze(dim=2)

#         # print('x_norm', x_norm)

#         d_x = (w_hat - cos_theta * x_hat) / x_norm
#         d_w = (x_hat - cos_theta * w_hat) / w_norm

#         # print('d_x', d_x[0, label[0], :10])

#         grad_x = ((cos_theta - one_hot) * d_x).sum(dim=1)
#         grad_w = ((cos_theta - one_hot) * d_w).sum(dim=0)
#         weight._grad = grad_w

#         # if grad_x.norm().mean() < 1e-5:
#         #     grad_x *= 10

#         cos_theta_y = torch.masked_select(cos_theta, one_hot.byte())
#         theta_y = torch.acos(cos_theta_y).mean() / math.pi   
#     return grad_x, theta_y


# class NormLinear(nn.Module):
    def __init__(self, feature_dim, class_number):
        super(NormLinear, self).__init__()
        self.feature_dim = feature_dim
        self.class_number = class_number
        self.weight = Parameter(torch.FloatTensor(class_number, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        output = cos_theta.clamp(-1, 1)
        # output = torch.acos(cos_theta)
        return output
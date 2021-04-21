from torch import nn
import torch.nn.functional as F
from utils import ReverseLayerF


def block(in_feat, out_feat, normalize=False):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, opt):
        super(FeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            *block(opt.domain_size, opt.yuce_dim_1, normalize=True),
            # *block(128, 256),
            # *block(256, 512),
            *block(opt.yuce_dim_1, opt.yuce_dim_2),
            nn.Linear(opt.yuce_dim_2, opt.latent_dim),
            nn.Tanh()
        )

    def forward(self, noise):
        feature = self.model(noise)
        # domain = domain.view(domain.size(0), opt.domain_size)
        return feature


# 类别分类器
class LableClassifier(nn.Module):
    def __init__(self, opt):
        super(LableClassifier, self).__init__()
        self.optget = opt
        self.model = nn.Sequential(
            *block(opt.latent_dim, opt.yuce_dim_2, normalize=True),
            # *block(128, 256),
            # *block(256, 512),
            #    *block(opt.yuce_dim_1, opt.yuce_dim_2),
            nn.Linear(opt.yuce_dim_2, opt.n_classes)
            # nn.Softmax()
        )

    def forward(self, features):
        feature = self.model(features)
        # domain = domain.view(domain.size(0), opt.domain_size)
        return F.softmax(feature, dim=self.optget.n_classes - 1)


# 域判别器
class DomainDiscriminator(nn.Module):
    def __init__(self, opt):
        super(DomainDiscriminator, self).__init__()

        self.model = nn.Sequential(
            *block(opt.latent_dim, opt.yuce_dim_1, normalize=True),
            *block(opt.yuce_dim_1, opt.yuce_dim_2),
            nn.Linear(opt.yuce_dim_2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_features, alpha):
        reversed_input = ReverseLayerF.apply(input_features, alpha)
        x = self.model(reversed_input)
        return x


# 特征生成器
class NormalGenerator(nn.Module):
    def __init__(self, opt):
        super(NormalGenerator, self).__init__()
        self.opt = opt
        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(opt.domain_size)),
            nn.Tanh()
        )

    def forward(self, z):
        domain = self.model(z)
        domain = domain.view(domain.size(0), self.opt.domain_size)
        return domain


# 特征监视器
class NormalDiscriminator(nn.Module):
    def __init__(self, opt):
        super(NormalDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(opt.domain_size), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        domain = z.view(z.size(0), -1)
        validity = self.model(domain)
        return validity


# 特征提取器
class FeatureExtractor0(nn.Module):
    def __init__(self, opt):
        super(FeatureExtractor0, self).__init__()
        self.model = nn.Sequential(nn.Linear(opt.domain_size, opt.latent_dim),
                                   nn.Sigmoid())

    def forward(self, noise):
        feature = self.model(noise)
        return feature


# 类别分类器
class LableClassifier0(nn.Module):
    def __init__(self, opt):
        super(LableClassifier0, self).__init__()
        self.optget = opt
        self.model = nn.Sequential(nn.Linear(opt.latent_dim, opt.n_classes))

    def forward(self, features):
        feature = self.model(features)
        return F.softmax(feature, dim=self.optget.n_classes - 1)


# 域判别器
class DomainDiscriminator0(nn.Module):
    def __init__(self, opt):
        super(DomainDiscriminator0, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 1),
            nn.Sigmoid())

    def forward(self, input_features, alpha):
        reversed_input = ReverseLayerF.apply(input_features, alpha)
        x = self.model(reversed_input)
        return x

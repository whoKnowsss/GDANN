import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Function
from sklearn.manifold import TSNE
import torch

import itertools
import os




class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer


def one_hot_embedding(labels, class_number=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      class_number: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(class_number)
    return y[labels]


def save_model(encoder, classifier, discriminator, training_mode, save_name):
    print('Save models ...')

    save_folder = 'trained_models'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(encoder.state_dict(), 'trained_models/encoder_' + str(training_mode) + '_' + str(save_name) + '.pt')
    torch.save(classifier.state_dict(), 'trained_models/classifier_' + str(training_mode) + '_' + str(save_name) + '.pt')

    if training_mode == 'dann':
        torch.save(discriminator.state_dict(), 'trained_models/discriminator_' + str(training_mode) + '_' + str(save_name) + '.pt')

    print('Model is saved !!!')


def plot_embedding(X, y, d, training_mode, save_name):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y = list(itertools.chain.from_iterable(y))
    y = np.asarray(y)

    plt.figure(figsize=(10, 10))
    for i in range(len(d)):  # X.shape[0] : 1024
        # plot colored number
        if d[i] == 0:
            colors = (0.0, 0.0, 1.0, 1.0)
        else:
            colors = (1.0, 0.0, 0.0, 1.0)
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=colors,
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if save_name is not None:
        plt.title(save_name)

    save_folder = 'saved_plot'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fig_name = 'saved_plot/' + str(training_mode) + '_' + str(save_name) + '.png'
    plt.savefig(fig_name)
    print('{} is saved'.format(fig_name))


# def visualize(encoder, training_mode, save_name):
#     # Draw 512 samples in test_data
#     source_test_loader = mnist.mnist_test_loader
#     target_test_loader = mnistm.mnistm_test_loader
#
#     # Get source_test samples
#     source_label_list = []
#     source_img_list = []
#     for i, test_data in enumerate(source_test_loader):
#         if i >= 16:  # to get only 512 samples
#             break
#         img, label = test_data
#         label = label.numpy()
#         img = img.cuda()
#         img = torch.cat((img, img, img), 1)  # MNIST channel 1 -> 3
#         source_label_list.append(label)
#         source_img_list.append(img)
#
#     source_img_list = torch.stack(source_img_list)
#     source_img_list = source_img_list.view(-1, 3, 28, 28)
#
#     # Get target_test samples
#     target_label_list = []
#     target_img_list = []
#     for i, test_data in enumerate(target_test_loader):
#         if i >= 16:
#             break
#         img, label = test_data
#         label = label.numpy()
#         img = img.cuda()
#         target_label_list.append(label)
#         target_img_list.append(img)
#
#     target_img_list = torch.stack(target_img_list)
#     target_img_list = target_img_list.view(-1, 3, 28, 28)
#
#     # Stack source_list + target_list
#     combined_label_list = source_label_list
#     combined_label_list.extend(target_label_list)
#     combined_img_list = torch.cat((source_img_list, target_img_list), 0)
#
#     source_domain_list = torch.zeros(512).type(torch.LongTensor)
#     target_domain_list = torch.ones(512).type(torch.LongTensor)
#     combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).cuda()
#
#     print("Extract features to draw T-SNE plot...")
#     combined_feature = encoder(combined_img_list)  # combined_feature : 1024,2352
#
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
#     dann_tsne = tsne.fit_transform(combined_feature.detach().cpu().numpy())
#
#     print('Draw plot ...')
#     save_name = save_name + '_' + str(training_mode)
#     plot_embedding(dann_tsne, combined_label_list, combined_domain_list, training_mode, save_name)


# def visualize_input():
#     source_test_loader = mnist.mnist_test_loader
#     target_test_loader = mnistm.mnistm_test_loader
#
#     # Get source_test samples
#     source_label_list = []
#     source_img_list = []
#     for i, test_data in enumerate(source_test_loader):
#         if i >= 16:  # to get only 512 samples
#             break
#         img, label = test_data
#         label = label.numpy()
#         img = img.cuda()
#         img = torch.cat((img, img, img), 1)  # MNIST channel 1 -> 3
#         source_label_list.append(label)
#         source_img_list.append(img)
#
#     source_img_list = torch.stack(source_img_list)
#     source_img_list = source_img_list.view(-1, 3, 28, 28)
#
#     # Get target_test samples
#     target_label_list = []
#     target_img_list = []
#     for i, test_data in enumerate(target_test_loader):
#         if i >= 16:
#             break
#         img, label = test_data
#         label = label.numpy()
#         img = img.cuda()
#         target_label_list.append(label)
#         target_img_list.append(img)
#
#     target_img_list = torch.stack(target_img_list)
#     target_img_list = target_img_list.view(-1, 3, 28, 28)
#
#     # Stack source_list + target_list
#     combined_label_list = source_label_list
#     combined_label_list.extend(target_label_list)
#     combined_img_list = torch.cat((source_img_list, target_img_list), 0)
#
#     source_domain_list = torch.zeros(512).type(torch.LongTensor)
#     target_domain_list = torch.ones(512).type(torch.LongTensor)
#     combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).cuda()
#
#     print("Extract features to draw T-SNE plot...")
#     combined_feature = combined_img_list  # combined_feature : 1024,3,28,28
#     combined_feature = combined_feature.view(1024, -1)  # flatten
#     # print(type(combined_feature), combined_feature.shape)
#
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
#     dann_tsne = tsne.fit_transform(combined_feature.detach().cpu().numpy())
#     print('Draw plot ...')
#     save_name = 'input_tsne_plot'
#     plot_embedding(dann_tsne, combined_label_list, combined_domain_list, 'input', 'mnist_n_mnistM')


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

import os

import scipy
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix
import scipy.io as scio
import numpy as np

from sklearn.preprocessing import StandardScaler
# from torch.utils.data import Dataset.Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def perf_ana(lable_true, lable_pred):
    cm = confusion_matrix(lable_true, lable_pred)
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Accuracy = (TP + TN) / (TP + FN + FP + TN)
    F1score = (2 * Precision * Recall) / (Precision + Recall)
    Specificity = TN / (TN + FP)

    Total = (TP + FN + FP + TN)
    Class1_C = (TP + FP)
    Class1_GT = (TP + FN)
    Class2_C = (TN + FN)
    Class2_GT = (FP + TN)
    ExpAcc = (((Class1_C * Class1_GT) / Total) + ((Class2_C * Class2_GT) / Total)) / Total
    Kappa = (Accuracy - ExpAcc) / (1 - ExpAcc)
    return Recall, Precision, Accuracy, F1score




def get_data(target_itr, domains_all,datapath):
    data_path_root = os.path.join(os.path.dirname(__file__), datapath)
    data_path = os.path.join(data_path_root, domains_all[target_itr] + '.mat')
    data = scio.loadmat(data_path)
    if datapath=='correctData':
        targetData = np.array(data['S'])
        targetLable = np.array(data['Slab'])
        sourceData = []
        sourceLable = []
        for one in domains_all:
            if one != domains_all[target_itr]:
                data_path = os.path.join(data_path_root, one + '.mat')
                data = scio.loadmat(data_path)
                if len(sourceData) == 0:
                    sourceData = data['S']
                    sourceLable = data['Slab']
                else:
                    sourceData = np.vstack((sourceData, data['S']))
                    sourceLable = np.vstack((sourceLable, data['Slab']))

        # 标准归一化
        std = StandardScaler()
        # # 将x进行标准化
        sourceData = std.fit_transform(sourceData)
        targetData = std.fit_transform(targetData)
        # if is_tensor:
        #     sourceData = torch.from_numpy(sourceData)
        #     targetData = torch.from_numpy(targetData)
        #     sourceLable = torch.from_numpy(sourceLable)
        #     targetLable = torch.from_numpy(targetLable)
        return sourceData, sourceLable, targetData, targetLable, domains_all[target_itr]
    else:
        targetData = np.array(data['data'])
        targetLable = np.array(data['label'].T)
        sourceData = []
        sourceLable = []
        for one in domains_all:
            if one != domains_all[target_itr]:
                data_path = os.path.join(data_path_root, one + '.mat')
                data = scio.loadmat(data_path)
                if len(sourceData) == 0:
                    sourceData = data['data']
                    sourceLable = data['label'].T
                else:
                    sourceData = np.vstack((sourceData, data['data']))
                    sourceLable = np.vstack((sourceLable, data['label'].T))

        # 标准归一化
        std = StandardScaler()
        # # 将x进行标准化
        sourceData = std.fit_transform(sourceData)
        targetData = std.fit_transform(targetData)
        # if is_tensor:
        #     sourceData = torch.from_numpy(sourceData)
        #     targetData = torch.from_numpy(targetData)
        #     sourceLable = torch.from_numpy(sourceLable)
        #     targetLable = torch.from_numpy(targetLable)
        return sourceData, sourceLable, targetData, targetLable, domains_all[target_itr]



def get_data_target_split_source(target_itr, domains_all,datapath):
    data_path_root = os.path.join(os.path.dirname(__file__), datapath)
    data_path = os.path.join(data_path_root, domains_all[target_itr] + '.mat')
    data = scio.loadmat(data_path)
    targetData = np.array(data['data'])
    targetLable = np.array(data['label'].T)
    sourceData = []
    sourceLable = []
    dictSourceList={}
    for one in domains_all:
        if one != domains_all[target_itr]:
            data_path = os.path.join(data_path_root, one + '.mat')
            data = scio.loadmat(data_path)
            dictSourceList[one]=[len(sourceData),len(sourceData)+len(targetData)]
            if len(sourceData) == 0:
                sourceData = data['data']
                sourceLable = data['label'].T
            else:
                sourceData = np.vstack((sourceData, data['data']))
                sourceLable = np.vstack((sourceLable, data['label'].T))

    # 标准归一化
    std = StandardScaler()
    # # 将x进行标准化
    sourceData = std.fit_transform(sourceData)
    targetData = std.fit_transform(targetData)
    return sourceData, sourceLable, targetData, targetLable, domains_all[target_itr],dictSourceList






def get_max_source(source_num,source_dict_list):
    # 获取列表的第二个元素
    def takeSecond(elem):
        return elem[1]
    # 指定第二个元素排序
    source_dict_list.sort(key=takeSecond,reverse = True)
    return [source_dict_list[i][0] for i in range(source_num)]

def get_max_source_list(source_dict_list):
    # 获取列表的第二个元素
    def takeSecond(elem):
        return elem[1]
    # 指定第二个元素排序
    source_dict_list.sort(key=takeSecond,reverse = True)
    return [source_dict_list[i][0] for i in range(len(source_dict_list))]

def get_data_one_and_one(source_itr,target_itr, domains_all):
    data_path_root = os.path.join(os.path.dirname(__file__), 'correctData')
    data_path = os.path.join(data_path_root, domains_all[target_itr] + '.mat')
    data = scio.loadmat(data_path)
    targetData = np.array(data['S'])
    targetLable = np.array(data['Slab'])
    data_path = os.path.join(data_path_root, domains_all[source_itr] + '.mat')
    data = scio.loadmat(data_path)
    sourceData = np.array(data['S'])
    sourceLable = np.array(data['Slab'])

    # 标准归一化
    std = StandardScaler()
    # # 将x进行标准化
    sourceData = std.fit_transform(sourceData)
    targetData = std.fit_transform(targetData)
    # if is_tensor:
    #     sourceData = torch.from_numpy(sourceData)
    #     targetData = torch.from_numpy(targetData)
    #     sourceLable = torch.from_numpy(sourceLable)
    #     targetLable = torch.from_numpy(targetLable)
    return sourceData, sourceLable, targetData, targetLable, domains_all[source_itr],domains_all[target_itr]

class DataLoaderData(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, Data, Label, Transform=None):
        self.Data = Data
        self.Label = Label
        self.Transform = Transform

    # 返回数据集大小
    def __len__(self):
        return len(self.Data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.IntTensor(self.Label[index])
        if self.Transform is not None:
            data = self.Transform(data)
        return data, label


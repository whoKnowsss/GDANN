import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data.dataset import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function

from model import FeatureExtractor, LableClassifier, DomainDiscriminator, NormalGenerator, \
    NormalDiscriminator
from utils import perf_ana, get_data_target_split_source, get_max_source_list

# 包含数据筛选的GDANN+正确率最大模型:
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch_two", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    # parser.add_argument("--batch_size_big", type=int, default=1024, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--domain_size", type=int, default=1647, help="size of each dimension")
    parser.add_argument("--yuce_dim_1", type=int, default=512, help="size of zhong jian layer 1")
    parser.add_argument("--yuce_dim_2", type=int, default=1024, help="size of zhong jian layer 2")
    parser.add_argument("--n_classes", type=int, default=2, help="size of zhong jian layer 2")
    parser.add_argument("--train_size", type=int, default=0.5, help="train rate of the target domain")
    parser.add_argument("--half_num", type=int, default=0.5, help="train rate of the target domain")
    parser.add_argument("--epoch_one", type=int, default=50, help="gan train epoch")
    # parser.add_argument("--source_subject", type=int, default=9, help="source subject num")
    parser.add_argument("--test_run_num", type=int, default=5, help="hou many times to run every number")

    parser.add_argument("--sourceSubjectNum", type=int, default=14, help="hou many source subjects num")
    # parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    # parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    opt = parser.parse_args()
    print(opt)

    # cpu_free = str(get_free_gpu())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Configure data loader
    # 创建DataLoader迭代器
    domains_all = ['MESMAR', 'ARCALE', 'SCAEMI', 'CILRAM', 'GNATN', 'CULLEO', 'BORGIA', 'VALNIC', 'DESTER', 'SALSTE',
                   'MARFRA', 'ANZALE', 'DIFANT', 'VALPAO', 'BORFRA']

    file_path = 'data_psd_0.5_alphabetagama'
    dictAllResult = {}

    for oneDomain in range(len(domains_all)):
        onePersonDict = {}
        [SourceData, SourceLable, TargetData, TargetLable, testName, dictSourceList] = get_data_target_split_source(
            oneDomain, domains_all, file_path)
        TargetDataTrain, TargetDataTest, TargetLableTrain, TargetLableTest = train_test_split(TargetData,
                                                                                              TargetLable,
                                                                                              test_size=1 - opt.train_size)

        TargetData, TargetLable, TargetDataTrain, TargetDataTest, TargetLableTrain, TargetLableTest = \
            torch.from_numpy(TargetData).float().to(device), torch.from_numpy(TargetLable).long().to(device), \
            torch.from_numpy(TargetDataTrain).float().to(device), torch.from_numpy(TargetDataTest).float().to(
                device), \
            torch.from_numpy(TargetLableTrain).long().to(device), torch.from_numpy(TargetLableTest).long().to(
                device)

        dataloaderTargetTrain = torch.utils.data.DataLoader(
            TensorDataset(TargetDataTrain, TargetLableTrain), shuffle=True,
            batch_size=opt.batch_size, drop_last=False)
        dataloaderTargetTestn = torch.utils.data.DataLoader(
            TensorDataset(TargetDataTest, TargetLableTest), shuffle=True,
            batch_size=opt.batch_size, drop_last=False)
        dataloaderTargetAll = torch.utils.data.DataLoader(
            TensorDataset(TargetData, TargetLable), shuffle=True,
            batch_size=opt.batch_size, drop_last=False)
        normal_genera = NormalGenerator(opt).to(device)
        normal_discrim = NormalDiscriminator(opt).to(device)
        # Optimizers
        optimizer_normal_G = torch.optim.Adam(normal_genera.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_normal_D = torch.optim.Adam(normal_discrim.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        # Loss function
        adversarial_normal_loss = torch.nn.BCELoss().to(device)

        # 预训练
        for epoch in range(opt.epoch_one):
            loss_all_d = 0
            loss_all_g = 0
            for i, (one_batch_X, _) in enumerate(dataloaderTargetTrain):
                # Adversarial ground truths
                valid = torch.FloatTensor(one_batch_X.size(0), 1).fill_(1.0).to(device)
                fake = torch.FloatTensor(one_batch_X.size(0), 1).fill_(0.0).to(device)

                # Configure input
                real_imgs = one_batch_X

                # -----------------
                #  Train Generator
                # -----------------
                # -----------------

                optimizer_normal_G.zero_grad()

                # Sample noise as generator input
                z = torch.from_numpy(np.random.normal(0, 1, (one_batch_X.shape[0], opt.latent_dim))).float().to(
                    device)

                # Generate a batch of images
                gen_imgs = normal_genera(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_normal_loss(normal_discrim(gen_imgs), valid)

                g_loss.backward()
                optimizer_normal_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_normal_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_normal_loss(normal_discrim(real_imgs), valid)
                fake_loss = adversarial_normal_loss(normal_discrim(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_normal_D.step()
                loss_all_g += g_loss.item()
                loss_all_d += d_loss.item()
            print("TargetName:[%s]-PreTrain: [Epoch %d/%d] [D loss: %f] [G loss: %f]"
                  % (domains_all[oneDomain], epoch, opt.epoch_one,
                     loss_all_d / len(dataloaderTargetTrain), loss_all_g / len(dataloaderTargetTrain)))

        # 源域数据筛选,获得最大list
        with torch.no_grad():
            listDict = []
            for key, value in dictSourceList.items():
                SourceDataOne = torch.from_numpy(SourceData).float().to(device)[value[0]:value[1]]
                SourcelabelOne = SourceLable[value[0]:value[1]]
                source_output = normal_discrim(SourceDataOne)
                listDict.append([key, (source_output).sum() / len(source_output)])
            listDictName = get_max_source_list(listDict)

        # 开始每个被试1~max_num的训练
        sourceSubjectNum = opt.sourceSubjectNum
        besttempAll = []
        for runTimes in range(1, opt.test_run_num):
            besttemp = []
            # Initialize generator and discriminator
            feature_ext = FeatureExtractor(opt).to(device)
            lable_cla = LableClassifier(opt).to(device)
            domain_dis = DomainDiscriminator(opt).to(device)

            # feature_loss = torch.nn.BCELoss()
            lable_loss = torch.nn.CrossEntropyLoss()
            domain_loss = torch.nn.BCELoss()

            # feature_loss.to(device)
            lable_loss.to(device)
            domain_loss.to(device)

            # 源域数据获取
            with torch.no_grad():
                SourceDataTemp = []
                SourceLabelTemp = []
                for dictOne in listDictName[0:sourceSubjectNum]:
                    if len(SourceDataTemp) == 0:
                        SourceDataTemp = SourceData[dictSourceList[dictOne][0]:dictSourceList[dictOne][1]]
                        SourceLabelTemp = SourceLable[dictSourceList[dictOne][0]:dictSourceList[dictOne][1]]
                    else:
                        SourceDataTemp = np.vstack(
                            (SourceDataTemp, SourceData[dictSourceList[dictOne][0]:dictSourceList[dictOne][1]]))
                        SourceLabelTemp = np.vstack(
                            (SourceLabelTemp, SourceLable[dictSourceList[dictOne][0]:dictSourceList[dictOne][1]]))
                SourceDataOne = torch.from_numpy(SourceDataTemp).float().to(device)
                SourceLableOne = torch.from_numpy(SourceLabelTemp).long().to(device)
                dataloaderSourceTrain = torch.utils.data.DataLoader(
                    TensorDataset(SourceDataOne, SourceLableOne), shuffle=True,
                    batch_size=opt.batch_size, drop_last=False)
                print(
                    "TargetName:[%s] SourceNum:[%d]-GetSourceSubject: [%s]"
                    % (domains_all[oneDomain], sourceSubjectNum, str(listDictName[0:sourceSubjectNum])))
            # ----------
            #  Training the Source Classifier
            # ----------
            normal_genera.eval()
            normal_discrim.eval()
            TargetDataTrainOne = torch.cat((TargetDataTrain, TargetDataTrain), 0)
            z = torch.from_numpy(
                np.random.normal(0, 1,
                                 (SourceDataOne.shape[0] - TargetDataTrainOne.shape[0], opt.latent_dim))).float().to(
                device)
            z = normal_genera(z)
            TargetDataTrainOne = torch.cat((TargetDataTrainOne, z.detach()), 0)
            TargetLableTrainOne = torch.ones(TargetDataTrainOne.shape[0], 1).float().to(device)

            dataloaderTargetTrainOne = torch.utils.data.DataLoader(
                TensorDataset(TargetDataTrainOne, TargetLableTrainOne), shuffle=True,
                batch_size=opt.batch_size, drop_last=False)

            feature_ext.train()
            lable_cla.train()
            domain_dis.train()
            optimizer = optim.SGD(
                [{'params': feature_ext.parameters()},
                 {'params': lable_cla.parameters()},
                 {'params': domain_dis.parameters()}],
                lr=0.005,
                momentum=0.9)

            # 正式训练
            for epoch in range(opt.epoch_two):
                start_steps = opt.epoch_two * len(dataloaderSourceTrain)
                total_steps = opt.epoch_two * len(dataloaderTargetTrainOne)
                loss_one_class = 0
                loss_one_total = 0
                for batch_idx, (source_data, target_data) in enumerate(
                        zip(dataloaderSourceTrain, dataloaderTargetTrainOne)):
                    source_image, source_label = source_data
                    target_image, target_label = target_data

                    p = float(batch_idx + start_steps) / total_steps
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                    # source_image = torch.cat((source_image, source_image, source_image), 1)

                    combined_image = torch.cat((source_image, target_image), 0)

                    optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
                    optimizer.zero_grad()

                    combined_feature = feature_ext(combined_image)
                    source_feature = feature_ext(source_image)

                    # 1.Classification loss
                    class_pred = lable_cla(source_feature)
                    class_loss = lable_loss(class_pred, source_label.squeeze())
                    loss_one_class += class_loss.item()
                    # 2. Domain loss
                    domain_pred = domain_dis(combined_feature, alpha)

                    domain_source_labels = torch.zeros(source_label.shape[0]).float().to(device)
                    domain_target_labels = torch.ones(target_label.shape[0]).float().to(device)
                    domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0)
                    domain_loss_ci = domain_loss(domain_pred.reshape(domain_pred.shape[0]), domain_combined_label)

                    total_loss = class_loss + domain_loss_ci
                    loss_one_total += total_loss.item()
                    total_loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    list = []
                    listTrue = []
                    for i, (one_batch_X, one_batch_Y) in enumerate(dataloaderTargetAll):
                        source_feature = feature_ext(one_batch_X)
                        source_output = lable_cla(source_feature)
                        source_output = source_output.data.max(1, keepdim=True)[1]
                        list.append([int(x) for y in source_output for x in y])
                        listTrue.append([int(x) for y in one_batch_Y for x in y])
                    temp_re_all = perf_ana([x for y in list for x in y], [x for y in listTrue for x in y])
                if besttemp == [] or besttemp[2] < temp_re_all[2]:
                    besttemp = temp_re_all

                print(
                    "TargetName:[%s] SourceNum:[%d]-Train: [runTime %d/%d] [Epoch %d/%d] [class loss: %f] [Total loss: %f] [accuracyNow: %f] [accuracyBest: %f]"
                    % (domains_all[oneDomain], sourceSubjectNum, runTimes, opt.test_run_num, epoch,
                       opt.epoch_two, loss_one_class / len(dataloaderSourceTrain),
                       loss_one_total / len(dataloaderSourceTrain), temp_re_all[2], besttemp[2])
                )

            # with torch.no_grad():
            #     list = []
            #     listTrue = []
            #     for i, (one_batch_X, one_batch_Y) in enumerate(dataloaderTargetAll):
            #         source_feature = feature_ext(one_batch_X)
            #         source_output = lable_cla(source_feature)
            #         source_output = source_output.data.max(1, keepdim=True)[1]
            #         list.append([int(x) for y in source_output for x in y])
            #         listTrue.append([int(x) for y in one_batch_Y for x in y])
            #     temp_re_all = perf_ana([x for y in list for x in y], [x for y in listTrue for x in y])
            #     print(
            #         "TargetName:[%s] SourceNum:[%d]-GetResult: [%s]"
            #         % (domains_all[oneDomain], sourceSubjectNum,  str(temp_re_all)))
            besttempAll.append(besttemp)
        onePersonDict = [besttempAll, listDictName[0:sourceSubjectNum]]
        dictAllResult[domains_all[oneDomain]] = onePersonDict
    print(dictAllResult)

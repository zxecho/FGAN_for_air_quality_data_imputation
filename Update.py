#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from util_tools import get_time_stamp, mkdir, sample_Z
from util_tools import loss_plot

clip_g = 1e-2
lambda_gp = 10


class LocalUpdate(object):
    def __init__(self, args, idx=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.idx = idx
        self.Dim = None
        self.trainX = None
        self.testX = None
        self.trainM = None
        self.testM = None
        self.Train_No = None
        self.Test_No = None
        self.p_hint = args.p_hint
        self.alpha = args.alpha

        self.cur_epoch = 0

    def train(self, G, D, dataset):

        mb_size = self.args.local_bs
        # local 参与方的优化器

        # optimizer_D = torch.optim.SGD(params=D.parameters(), lr=self.args.fed_d_lr)
        # optimizer_G = torch.optim.SGD(params=G.parameters(), lr=self.args.fed_g_lr)

        optimizer_D = torch.optim.Adam(params=D.parameters(), lr=self.args.fed_d_lr)
        optimizer_G = torch.optim.Adam(params=G.parameters(), lr=self.args.fed_g_lr)
        if self.args.use_amp:
            # 使用amp混合精度
            scaler_D = GradScaler()
            scaler_G = GradScaler()

        # optimizer_D = torch.optim.RMSprop(params=D.parameters(), lr=self.args.fed_d_lr)
        # optimizer_G = torch.optim.RMSprop(params=G.parameters(), lr=self.args.fed_g_lr)

        # 用于记录每个epoch的loss信息
        # G_epoch_loss = []
        # D_epoch_loss = []
        # 记录存储每个batch的loss信息
        G_batch_loss = []
        D_batch_loss = []
        # 用于记录G的生成效果
        G_MSE_batch_trian_loss = []
        # G_MSE_epoch_train_loss = []
        G_MSE_batch_test_loss = []
        # G_MSE_epoch_tets_loss = []

        self.Dim = dataset['d']
        self.trainX = dataset['train_x']
        self.testX = dataset['test_x']
        self.trainM = dataset['train_m']
        self.testM = dataset['test_m']
        self.Train_No = dataset['train_no']
        self.Test_No = dataset['test_no']

        for j in range(self.args.local_ep):
            mb_idx = self.sample_idx(self.Train_No, mb_size)
            X_mb = self.trainX[mb_idx, :]

            Z_mb = sample_Z(mb_size, self.Dim)
            M_mb = self.trainM[mb_idx, :]

            if self.p_hint == 1.0:
                H_mb = M_mb
            else:
                H_mb1 = self.sample_M(mb_size, self.Dim, 1 - self.p_hint)
                H_mb = M_mb * H_mb1

            New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

            X_mb = torch.tensor(X_mb, device="cuda", dtype=torch.float32)
            M_mb = torch.tensor(M_mb, device="cuda", dtype=torch.float32)
            H_mb = torch.tensor(H_mb, device="cuda", dtype=torch.float32)
            New_X_mb = torch.tensor(New_X_mb, device="cuda", dtype=torch.float32)

            if self.args.use_amp:
                optimizer_D.zero_grad()
                with autocast():
                    D_loss_curr = self.compute_D_loss(G, D, M=M_mb, New_X=New_X_mb, H=H_mb)
                scaler_D.scale(D_loss_curr).backward()
                scaler_D.step(optimizer_D)
                scaler_D.update()

                if j % self.args.fed_n_critic == 0:
                    optimizer_G.zero_grad()
                    with autocast():
                        G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = self.compute_G_loss(G, D, X=X_mb, M=M_mb,
                                                                                                   New_X=New_X_mb,
                                                                                                   H=H_mb)
                    scaler_G.scale(G_loss_curr).backward()
                    scaler_G.step(optimizer_G)
                    scaler_G.update()

            else:
                optimizer_D.zero_grad()
                D_loss_curr = self.compute_D_loss(G, D, M=M_mb, New_X=New_X_mb, H=H_mb)
                # 计算WGAN-D的损失函数
                # D_loss_curr = self.compute_WD_loss(G, D, M=M_mb, New_X=New_X_mb, H=H_mb)
                D_loss_curr.backward()
                optimizer_D.step()

                if j % self.args.fed_n_critic == 0:
                    optimizer_G.zero_grad()
                    G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = self.compute_G_loss(G, D, X=X_mb, M=M_mb,
                                                                                               New_X=New_X_mb, H=H_mb)

                    G_loss_curr.backward()
                    optimizer_G.step()

            if self.args.verbose and j % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\t G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                    self.args.local_stations[self.idx], j, self.args.local_ep,
                    100. * j / self.args.local_ep,
                    G_loss_curr,
                    D_loss_curr
                ))
            G_batch_loss.append(G_loss_curr)
            D_batch_loss.append(D_loss_curr)
            G_MSE_batch_trian_loss.append(MSE_train_loss_curr)
            G_MSE_batch_test_loss.append(MSE_test_loss_curr)

        G_epoch_loss_mean = sum(G_batch_loss) / len(G_batch_loss)
        D_epoch_loss_mean = sum(D_batch_loss) / len(D_batch_loss)
        G_MSE_epoch_train_loss_mean = sum(G_MSE_batch_trian_loss) / len(G_MSE_batch_trian_loss)
        G_MSE_epoch_test_loss_mean = sum(G_MSE_batch_test_loss) / len(G_MSE_batch_test_loss)
        return G.state_dict(), D.state_dict(), G_epoch_loss_mean, D_epoch_loss_mean, \
               G_MSE_epoch_train_loss_mean, G_MSE_epoch_test_loss_mean, self.Train_No

    def independent_training(self, G, D, dataset, station_name='', save_pth_pre=''):
        # 写入训练过程数据
        fw_name = save_pth_pre + 'indpt_' + station_name + '_log.txt'
        fw_fed_main = open(fw_name, 'w+')
        fw_fed_main.write('iter\t G_loss\t D_loss\t G_train_MSE_loss\t G_test_MSE_loss\t \n')

        mb_size = self.args.local_bs
        # local 参与方的优化器
        # optimizer_D = torch.optim.SGD(params=D.parameters(), lr=self.args.d_lr, momentum=0.9)
        # optimizer_G = torch.optim.SGD(params=G.parameters(), lr=self.args.g_lr, momentum=0.9)

        optimizer_D = torch.optim.Adam(params=D.parameters(), lr=self.args.d_lr)
        optimizer_G = torch.optim.Adam(params=G.parameters(), lr=self.args.g_lr)

        if self.args.use_amp:
            # 使用amp混合精度
            scaler_D = GradScaler()
            scaler_G = GradScaler()

        # optimizer_D = torch.optim.RMSprop(params=D.parameters(), lr=self.args.d_lr)
        # optimizer_G = torch.optim.RMSprop(params=G.parameters(), lr=self.args.g_lr)

        # 用于调整学习率
        if self.args.lr_decay:
            D_StepLR = torch.optim.lr_scheduler.StepLR(optimizer_D,
                                                       step_size=self.args.d_lr_decay_step,
                                                       gamma=self.args.d_lr_decay)
            G_StepLR = torch.optim.lr_scheduler.StepLR(optimizer_G,
                                                       step_size=self.args.g_lr_decay_step,
                                                       gamma=self.args.g_lr_decay)
            # D_StepLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1, patience=10,
            #                                                       verbose=True, threshold=0.0001,
            #                                                       threshold_mode='rel',
            #                                                       min_lr=0)
            # G_StepLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=10,
            #                                                       verbose=True, threshold=0.0001,
            #                                                       threshold_mode='rel',
            #                                                       min_lr=0)

        # 用于绘图
        fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

        # 用于记录每个epoch的loss信息
        G_epoch_loss = []
        D_epoch_loss = []
        # 记录存储每个batch的loss信息
        # G_batch_loss = []
        # D_batch_loss = []
        # 用于记录G的生成效果
        # G_MSE_batch_trian_loss = []
        G_MSE_epoch_train_loss = []
        # G_MSE_batch_test_loss = []
        G_MSE_epoch_tets_loss = []

        self.Dim = dataset['d']
        self.trainX = dataset['train_x']
        self.testX = dataset['test_x']
        self.trainM = dataset['train_m']
        self.testM = dataset['test_m']
        self.Train_No = dataset['train_no']
        self.Test_No = dataset['test_no']

        print('Station ' + station_name + ' is under Training...')
        # 这次不加上这个 self.args.local_ep
        with tqdm(range(self.args.epochs)) as tq:
            for j in tq:  # 暂时取消self.args.local_ep *
                self.cur_epoch = j
                tq.set_description('Local Updating')
                mb_idx = self.sample_idx(self.Train_No, mb_size)
                X_mb = self.trainX[mb_idx, :]

                Z_mb = sample_Z(mb_size, self.args.input_dim) * 0.1
                M_mb = self.trainM[mb_idx, :]

                # 当p_hint=1时，即为原始的condition gan；否则，即为GAIN算法
                if self.p_hint == 1.0:
                    H_mb = M_mb
                else:
                    H_mb1 = self.sample_M(mb_size, self.Dim, 1 - self.p_hint)
                    H_mb = M_mb * H_mb1

                New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

                X_mb = torch.tensor(X_mb, device=self.args.device, dtype=torch.float32)
                M_mb = torch.tensor(M_mb, device=self.args.device, dtype=torch.float32)
                H_mb = torch.tensor(H_mb, device=self.args.device, dtype=torch.float32)
                New_X_mb = torch.tensor(New_X_mb, device=self.args.device, dtype=torch.float32)

                if self.args.use_amp:
                    optimizer_D.zero_grad()
                    with autocast():
                        D_loss_curr = self.compute_D_loss(G, D, M=M_mb, New_X=New_X_mb, H=H_mb)
                    scaler_D.scale(D_loss_curr).backward()
                    scaler_D.step(optimizer_D)
                    scaler_D.update()

                    if j % self.args.n_critic == 0:
                        optimizer_G.zero_grad()
                        with autocast():
                            G_loss_curr, MSE_train_loss_curr, RMSE_test_loss_curr = self.compute_G_loss(G, D,
                                                                                                        X=X_mb,
                                                                                                        M=M_mb,
                                                                                                        New_X=New_X_mb,
                                                                                                        H=H_mb)
                        scaler_G.scale(G_loss_curr).backward()
                        scaler_G.step(optimizer_D)
                        scaler_G.update()
                else:
                    optimizer_D.zero_grad()
                    D_loss_curr = self.compute_D_loss(G, D, M=M_mb, New_X=New_X_mb, H=H_mb)

                    D_loss_curr.backward()
                    optimizer_D.step()

                    if j % self.args.n_critic == 0:
                        optimizer_G.zero_grad()
                        G_loss_curr, MSE_train_loss_curr, RMSE_test_loss_curr = self.compute_G_loss(G, D, X=X_mb,
                                                                                                    M=M_mb,
                                                                                                    New_X=New_X_mb,
                                                                                                    H=H_mb)

                        G_loss_curr.backward()
                        optimizer_G.step()
                if self.args.lr_decay:
                    D_StepLR.step()
                    G_StepLR.step()

                # 保存模型的文件夹名称
                # file_name = save_pth_pre.split('/')[2]
                file_name = save_pth_pre.split('/')
                file_name = file_name[2] + '/' + file_name[3]
                # %% Intermediate Losses
                tq.set_postfix(Train_MSE_loss=np.sqrt(MSE_train_loss_curr.item()),
                               Train_RMSE=np.sqrt(RMSE_test_loss_curr.item()))
                # 保存模型
                if j % 100 == 0:
                    self.save_model(G, D, file_name, station_name)

                # 调整学习率 步骤
                # G_StepLR.step()
                # D_StepLR.step()

                if j % 1 == 0:
                    # 保存训练过程数据
                    G_epoch_loss.append(G_loss_curr)
                    D_epoch_loss.append(D_loss_curr)

                    G_MSE_epoch_train_loss.append(MSE_train_loss_curr)
                    G_MSE_epoch_tets_loss.append(RMSE_test_loss_curr)

                    fw_fed_main.write('{}\t {:.5f}\t {:.5f}\t {:.5f}\t {:.5f}\t \n'.format(j, G_loss_curr, D_loss_curr,
                                                                                           MSE_train_loss_curr,
                                                                                           RMSE_test_loss_curr))

        # self.plot_progess_info(axs[0, 0], G_epoch_loss, 'G loss')
        # self.plot_progess_info(axs[0, 1], D_epoch_loss, 'D loss')
        # self.plot_progess_info(axs[1, 0], G_MSE_epoch_train_loss, 'G MSE training loss')
        # self.plot_progess_info(axs[1, 1], G_MSE_epoch_tets_loss, 'RMSE on training dataset')
        # Print the loss info in training progress
        loss_plot(axs[0, 0], G_epoch_loss, 'G loss')
        loss_plot(axs[0, 1], D_epoch_loss, 'D loss')
        loss_plot(axs[1, 0], G_MSE_epoch_train_loss, 'G MSE training loss')
        loss_plot(axs[1, 1], G_MSE_epoch_tets_loss, 'RMSE on training dataset')

        plt.savefig(save_pth_pre + 'indpt_{}_info.eps'.format(station_name))
        plt.savefig(save_pth_pre + 'indpt_{}_info.svg'.format(station_name))

        plt.close()

        return G

    def compute_D_loss(self, G, D, M, New_X, H):
        """
        :arg
        M: mask
        New_X: 缺失后的数据
        H： hint
        """
        # 生成器
        G_sample = G(New_X, M)
        # 将生成的数据与原数据进行组合，相当于用生成的数据填充原来空缺的地方
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # 判别器
        D_prob = D(Hat_New_X, H)
        # 使用CGAN
        # D_prob = D(Hat_New_X, M)

        # 损失函数
        D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob))
        # D_loss = -torch.mean(M * torch.log(D_prob) - (1 - M) * torch.log(D_prob))
        return D_loss

    def compute_G_loss(self, G, D, X, M, New_X, H):
        """
        :param X:  原数据
        :param M: Mask
        :param New_X: 原数据进行人为缺失和加上扰动
        :param H: Hint
        :return:
        """
        # %% Structure
        # Generator
        G_sample = G(New_X, M)

        # Combine with original data
        Hat_New_X = New_X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = D(Hat_New_X, H)
        # 使用CGAN
        # D_prob = D(Hat_New_X, M)

        # %% Loss
        G_loss1 = -torch.mean((1 - M) * torch.log(D_prob))
        MSE_train_loss = torch.mean((M * New_X - M * G_sample) ** 2) / torch.mean(M)
        # RMSE_train_loss = torch.sqrt(MSE_train_loss)

        alpha = self.alpha
        if (self.cur_epoch + 1) % self.args.d_lr_decay_step == 0:
            alpha = self.alpha * 0.9

        G_loss = G_loss1 + alpha * MSE_train_loss

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
        RMSE_test_loss = torch.sqrt(MSE_test_loss)
        return G_loss, self.alpha * MSE_train_loss, RMSE_test_loss

    def sample_M(self, m, n, p):
        """
        m: batch_size
        n: Dim 数据的属性维度
        p: 采样概率
        """
        # Hint vector的生成函数
        A = np.random.uniform(0., 1., size=[m, n])
        B = A > p
        C = 1. * B
        return C

    def sample_idx(self, m, n):
        """
        :arg
        n: 采样index的个数
        m: 样本总个数
        """
        A = np.random.permutation(m)
        idx = A[:n]
        return idx

    def Algo_test(self, G, X, M, New_X):
        """
        用于测试模型
        :param X: 原数据
        :param M: mask
        :param New_X: 构造出的包含Missing的数据
        :return:
        """
        # %% Structure
        # Generator
        G_sample = G(New_X, M)

        # %% MSE Performance metric
        MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
        return MSE_test_loss, G_sample

    def testing(self, G, mode_name=''):
        # 写入测试过程数据
        # fw_name = './results/indpt_test_' + mode_name + '_' + get_time_stamp() + '_log.txt'
        # fw_test = open(fw_name, 'w+')

        # 对算法进行测试
        Z_mb = sample_Z(self.Test_No, self.Dim)
        M_mb = self.testM
        X_mb = self.testX

        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

        X_mb = torch.tensor(X_mb, device='cuda', dtype=torch.float32)
        M_mb = torch.tensor(M_mb, device='cuda', dtype=torch.float32)
        New_X_mb = torch.tensor(New_X_mb, device='cuda', dtype=torch.float32)

        MSE_final, Sample = self.Algo_test(G, X=X_mb, M=M_mb, New_X=New_X_mb)

        print('Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))

        # imputed_data = M_mb * X_mb + (1 - M_mb) * Sample
        # print("Imputed test data:")
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})
        # print(imputed_data.cpu().detach().numpy())
        # 写入数据
        # fw_test.write('{} MSE Test : {}'.format(mode_name, MSE_final))

        return MSE_final

    def save_model(self, G, D, save_file='', station_name=''):
        mkdir('./saved_model/' + save_file + '/')
        torch.save(D.state_dict(), './saved_model/' + save_file + '/independent_' + station_name + '_D.pkl')
        torch.save(G.state_dict(), './saved_model/' + save_file + '/independent_' + station_name + '_G.pkl')

    def plot_progess_info(self, axs, loss_data, name=None):
        axs.plot(range(len(loss_data)), loss_data, label=name)
        axs.tick_params(label_size=13)
        axs.set_ylabel(name, size=13)
        axs.set_xlabel('Epochs', size=13)
        axs.legend()
        # axs.set_title(name)

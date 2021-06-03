#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # 数据集相关的参数
    parser.add_argument('--local_stations', type=list, default=['A', 'B', 'C', 'D', 'E', 'F'], help='所有气象站点')
    parser.add_argument('--select_dim', type=list, default=['PM2_5', 'PM10', 'SO2', 'CO', 'O3', 'NOX'],
                        help='选择的气象数据属性')
    parser.add_argument('--selected_stations', type=list, default=['A', 'A', 'A'],
                        help='所有气象站点')
    parser.add_argument('--clients', type=list, default=['P1', 'P2', 'P3'], help='所有参与方')
    parser.add_argument('--sample_interval', type=list, default=[3, 2, 1],
                        help='气象站收集数据时间间隔(h)')
    # 参与方的编号 The number of participants
    parser.add_argument('--num_d', type=int, default=[1, 2, 3], help="number of devices for each participant: K")
    parser.add_argument('--missing_ratios', type=list, default=[0.05, 0.10, 0.15],
                        help='按照选择的气象站点顺序，设定每个站点数据的缺失比例')
    parser.add_argument('--train_numbers', type=list, default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        help='按照选择的气象站点顺序，设定每个站点数据训练数据比例')
    parser.add_argument('--load_numbers', type=list, default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        help='按照选择的气象站点顺序，设定每个站点数据载入数据量')
    # parser.add_argument('--num_p', type=int, default=2, help="number of participants: Np")
    parser.add_argument('--quarter_index', type=list, default=[[1, 2, 3, 4],
                                                               [1, 2, 3, 4],
                                                               [1, 2, 3, 4],
                                                               [1, 2, 3, 4],
                                                               [1, 2, 3, 4],
                                                               [1, 2, 3, 4]],
                        help='按照选择的季节和气象站点顺序，设定每个站点数据载入数据')
    parser.add_argument('--missing_rate', type=float, default=0.3, help="number of classes")
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--dataset_path', type=str, default='', help="path of saved dataset file")

    # federated arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--idpt_epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=3, help="number of users: K")
    parser.add_argument('--weights_avg', type=bool, default=True, help="if use weights avg")
    parser.add_argument('--wa_type', type=str, default='missing_number',
                        help="weight average according missing_ratio or missing_number")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--fed_n_critic', type=int, default=3, help='the n_criitic of FGAN')
    #  ========= optimizer 相关 ==========
    parser.add_argument('--lr_decay', type=bool, default=True, help="IF using learning rate decay")
    parser.add_argument('--g_lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--d_lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--g_lr_decay', type=float, default=0.9, help="G learning rate decay rate")
    parser.add_argument('--g_lr_decay_step', type=int, default=200, help="learning rate")
    parser.add_argument('--d_lr_decay', type=float, default=0.9, help="D learning rate decay rate")
    parser.add_argument('--d_lr_decay_step', type=int, default=200, help="D learning rate decay after N step")
    # 专门用于联邦的生成对抗网络
    parser.add_argument('--fed_g_lr', type=float, default=1e-4, help="Fed G learning rate")
    parser.add_argument('--fed_d_lr', type=float, default=1e-4, help="Fed D learning rate")
    parser.add_argument('--fed_g_lr_decay', type=float, default=0.9, help="Fed G learning rate decay rate")
    parser.add_argument('--fed_g_lr_decay_step', type=int, default=200, help="Fed learning rate")
    parser.add_argument('--fed_lambda_term', type=float, default=20, help='the guassian clip lambda value of WGAN')

    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # independent training arguments
    parser.add_argument('--independent_usrs_training', type=bool, default=True, help="if independent_usrs_training")

    # CGAIN model arguments
    parser.add_argument('--input_dim', type=int, default=128, help="生成网络输入随机噪声层的单元数量")
    parser.add_argument('--G_hidden_dim', type=int, default=64, help="生成网络隐藏层的单元数量")
    parser.add_argument('--D_hidden_dim', type=int, default=64, help="判别网络隐藏层的单元数量")
    parser.add_argument('--model', type=str, default='FCGAI', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    # 算法参数
    parser.add_argument('--p_hint', type=float, default=0.9, help="GAIN 模型D 网络的提示率")
    parser.add_argument('--alpha', type=float, default=10, help="D 损失函数的超参数")
    # 用于W-GAN
    parser.add_argument('--clip_gradient', type=str, default=1e-2, help='WGAN model name')
    parser.add_argument('--clip_value', type=float, default=0.1, help='the gradient clip value of WGAN')
    parser.add_argument('--gan_categories', type=str, default='wGAN', help='the class of GAN')
    parser.add_argument('--lambda_term', type=float, default=10, help='the guassian clip lambda value of WGAN')
    parser.add_argument('--n_critic', type=int, default=5, help='the n_criitic of WGAN')
    # other arguments
    parser.add_argument('--use_amp', action='store_true', help='whether AUTOMATIC MIXED PRECISION(混合精度) or not')
    parser.add_argument('--use_saved_model', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
    args = parser.parse_args()
    return args

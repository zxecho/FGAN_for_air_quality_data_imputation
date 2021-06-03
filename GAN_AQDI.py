#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
import copy
from copy import deepcopy
import numpy as np
import json
import torch

from param_options import args_parser
# from LoadandCreateDatasets import get_saved_datasets
from CreateAndLoadDatasets import get_saved_datasets, get_saved_datasets_vall
from util_tools import get_time_stamp, mkdir, save_model, norm, compute_avg_of_data_in_file
import matplotlib.pyplot as plt
from GAN_AQDI_test_run import gain_test_exp, plot_indicator_avg_results_m
from plot_indexes_resluts import plot_indicator_results
from Fed_GAIN.show_dataset import plot_fed_avg_acc
# from Fed_GAIN.GAN_training import GANUpdate as LocalUpdate
# from Update import LocalUpdate  # 原始的GAN
from GAN_training import GANUpdate as LocalUpdate
from GAIN_model import Generator, Discriminator, weights_init


matplotlib.use('Agg')


def loss_plot(axs, loss_data, name=None):
    axs.plot(range(len(loss_data)), loss_data)
    axs.set_title(name)


def GAIN_main(args, save_path=''):
    # parse args
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    data_dim = len(args.select_dim)  # 真实数据维度，即生成器输出
    input_dim = args.input_dim  # 输入随机向量维度
    G_H_Dim = args.G_hidden_dim  # 设置G隐藏层的网络单元数量
    D_H_Dim = args.G_hidden_dim  # 设置D隐藏层的网络单元数量

    # 选择的站点
    stations = args.selected_stations

    # 载入数据并进行构造
    # 载入所有选定站点的数据集
    all_station_datasets = get_saved_datasets(args)
    # all_station_datasets = get_saved_datasets_vall(args)

    # 所有参与方进行独立的本地训练
    print('Start independent training!\n')
    # 用于记录indpendent local usr测试集结果
    num_p = len(args.selected_stations)
    num_d = args.num_d
    for p, d, dataset, station in zip(range(num_p), num_d,
                                      all_station_datasets, stations):
        # 建立GAIN model
        # 新建生成器网络G
        G = Generator(input_dim, G_H_Dim, data_dim).to(args.device)
        # 新建判别器网络D
        D = Discriminator(data_dim, D_H_Dim).to(args.device)
        print('Generater network :\n', G)
        print('Discriminator network :\n', D)

        G.apply(weights_init)
        D.apply(weights_init)

        station_name = station + '{}{}'.format(p, d)
        local = LocalUpdate(args=args)
        local_g = local.independent_training(G, D, dataset, station_name, save_path)

        # 清除之前独立学习的主循环所占用的显存空间
        torch.cuda.empty_cache()
    plt.cla()  # 清除之前绘图
    plt.close()


if __name__ == '__main__':
    args = args_parser()

    # 做实验
    exp_total_time = 3
    cross_validation_sets = 5

    # dataset_number = 'one_mi((A10)_1)'
    # dataset_number = 'one_mi_v1((A5)_1)'

    results_saved_file = 'results_one_dn'
    results_plot_file = 'plot_results_one_dn'

    indicator_list = ['rmse', 'd2', 'r2', 'all_rmse']
    model_name_list = ['idpt']

    params_test_list = [15, 20, 25, 30, 35, 40]
    test_param_name = 'missing_ratio'

    # params_test_list = [10]
    # test_param_name = 'alpha'

    for param in params_test_list:

        print('**  {} params test: {}  **'.format(test_param_name, param))
        dataset_number = '(A{})_normCO_1r_1P'.format(param)
        # dataset_number = 'one_mi((A{})_1)'.format(param)
        exp_name = 'GANI_{}'.format(dataset_number)
        # args.alpha = param
        # exp_name = 'Test{}_P09D3T3_nc1a{}_GAIN_{}'.format(get_time_stamp(), param, dataset_number)
        ex_params_settings = {
            'algo_name': 'GANI_AQDI',
            'dataset': '{}'.format(dataset_number),
            'datasets_number': cross_validation_sets,
            'epochs': args.epochs,
            'local_ep': args.local_ep,
            'local_bacthsize': args.local_bs,
            'n_critic': args.n_critic,
            'G_hidden_n': args.G_hidden_dim,
            'D_hidden_n': args.D_hidden_dim,
            'activate_function': 'ReLU',
            'optimizer': 'Adam',
            'idpt_d_lr': args.d_lr,
            'idpt_g_lr': args.g_lr,
            'lr_decay': args.lr_decay,
            'lr_decay_rate': args.g_lr_decay,
            'lr_decay_step': args.g_lr_decay_step,
            'clip_value': args.clip_value,
            'p_hint': args.p_hint,
            'alpha': args.alpha
        }

        # 存储主文件路径
        result_save_main_file = './{}/'.format(results_saved_file) + exp_name + '/'
        mkdir(result_save_main_file)

        # 保存参数配置
        params_save_name = result_save_main_file + 'params_settings.json'
        with open(params_save_name, 'w+') as jsonf:
            json.dump(ex_params_settings, jsonf)

        for i in range(cross_validation_sets):
            print('============= Start training at datasets {} =============='.format(i))
            # 用于统计各种指标，建立相对应的文件夹
            result_save_file = './{}/'.format(results_saved_file) + exp_name + '/datasets_{}/'.format(i)

            for index in indicator_list:
                for model_name in model_name_list:
                    test_result_save_path = result_save_file + index + '/' + model_name
                    mkdir(test_result_save_path)

            for exp_t in range(exp_total_time):
                # 当前数据集
                args.dataset_path = './constructed_datasets/{}/{}/'.format(dataset_number, i)

                print('******* Training epoch {} *******'.format(exp_t))
                save_path_pre = result_save_file + str(exp_t) + '/'
                mkdir(save_path_pre)
                GAIN_main(args, save_path_pre)
                gain_test_exp(args, result_save_file, exp_t)

        # 清空GPU缓存
        torch.cuda.empty_cache()

        print('>>> Finished current training & testing!')

        result_save_root = './{}/'.format(results_saved_file) + exp_name + '/'
        plots_save_root = './{}/'.format(results_plot_file) + exp_name + '/'
        # indicator_name = 'all_rmse'
        leg = ['Local']

        for indicator_name in indicator_list:
            # 建立保存结果的文件夹
            indicator_avg_results_csv_save_fpth = result_save_root + 'avg_' + indicator_name + '/'

            for mode in leg:
                diff_indicator_avg_result_save_dir = indicator_avg_results_csv_save_fpth + mode + '/'
                mkdir(diff_indicator_avg_result_save_dir)

            # 计算每个数据集的几次实验的均值
            for c in range(cross_validation_sets):
                results_logdir = [
                    result_save_root + 'datasets_' + str(c) + '/' + indicator_name + '/' + model_name + '/'
                    for model_name in model_name_list]

                compute_avg_of_data_in_file(args, c, results_logdir, indicator_avg_results_csv_save_fpth,
                                            indicator_name, leg)

            # 绘制测试结果图像
            print('\033[0;34;40m [Visulize] Save every component indicator result \033[0m')

            print('[Visulize] {} results'.format(indicator_name))

            results_logdir = [result_save_root + 'avg_' + indicator_name + '/' + model_name + '/' for model_name in leg]
            fig_save_path = plots_save_root + indicator_name + '/'
            csv_save_fpth = result_save_root + 'avg_' + indicator_name + '/'
            mkdir(fig_save_path)

            # 将每个数据集计算的均值再计算总的均值和绘制方差均值线图
            # plot_indicator_avg_results_m(results_logdir, fig_save_path, 'station',
            #                              indicator_name, csv_save_fpth, leg=leg)
            plot_fed_avg_acc(results_logdir, indicator_name, fig_save_path)
            # plot_indicator_results(results_logdir, fig_save_path, indicator_name, leg=leg)

            print("\033[0;34;40m >>>[Visulize] Finished save {} resluts figures! \033[0m".format(indicator_name))

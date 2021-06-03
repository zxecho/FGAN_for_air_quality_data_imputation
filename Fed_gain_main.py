#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
import json
import torch
from tqdm import tqdm

from param_options import args_parser
from FedAvg import FedAvg, FedWeightedAvg
# from LoadandCreateDatasets import get_saved_datasets
from CreateAndLoadDatasets import get_saved_datasets
from util_tools import get_time_stamp, mkdir, save_model, norm, compute_avg_of_data_in_file, loss_plot
from Fed_gain_test_run import fed_gain_test_exp, compute_indicator_results
from Fed_GAIN.show_dataset import plot_indicator_avg_results_m
from plot_indexes_resluts import plot_indicator_results

matplotlib.use('Agg')


def fed_main(args, save_path=''):
    if args.gan_categories == 'wGAN':
        # W-GAN
        from WAGIN_model import Generator, Discriminator, weights_init
        from WGAN_LocalUpdate import GANUpdate as LocalUpdate
    else:
        # 原始的GAN
        from Update import LocalUpdate
        from GAIN_model import Generator, Discriminator, weights_init
    # parse args
    args.device = torch.device(torch.device('cuda') if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    data_dim = len(args.select_dim)  # 真实数据维度，即生成器输出
    input_dim = args.input_dim  # 输入随机向量维度

    G_H_Dim = args.G_hidden_dim  # 设置G隐藏层的网络单元数量
    D_H_Dim = args.G_hidden_dim  # 设置D隐藏层的网络单元数量

    # 选择的站点
    stations = args.selected_stations

    # 载入数据并进行构造
    # 载入所有选定站点的数据集
    all_station_datasets = get_saved_datasets(args)

    # 建立GAIN model
    # 新建生成器网络G
    G = Generator(input_dim, G_H_Dim, data_dim).to(args.device)
    # 新建判别器网络D
    D = Discriminator(data_dim, D_H_Dim).to(args.device)

    G.apply(weights_init)
    D.apply(weights_init)

    # 建立每个客户的独立个体
    independent_usrs_G = []
    independent_usrs_D = []

    for _ in range(len(stations)):
        # 为每个站点新建一个独立的个体网络
        independent_G = deepcopy(G)
        independent_D = deepcopy(D)
        independent_usrs_G.append(independent_G)
        independent_usrs_D.append(independent_D)

    print('Generater network :\n', G)
    print('Discriminator network :\n', D)

    G.train()  # 生成网络
    D.train()  # 辨别网络

    # 拷贝网络参数
    g_w_glob = G.state_dict()
    d_w_glob = D.state_dict()

    # training
    # 用于记录训练过程数据
    g_loss_train, d_loss_train = [], []
    g_mse_train_loss_avg_l = []
    g_mse_test_loss_avg_l = []

    # 所有参与方进行独立的本地训练
    if args.independent_usrs_training:
        print('Start independent training!\n')
        # 用于记录indpendent local usr测试集结果
        local_mse_test_l = []
        num_p = len(args.selected_stations)
        num_d = args.num_d
        for p, d, idp_G, idp_D, dataset, station in zip(range(num_p), num_d, independent_usrs_G, independent_usrs_D,
                                                        all_station_datasets, stations):
            # 针对load_dataset_v2，对不同参与方，每个参与方都有自己的站点数据添加
            if type(station) == list:
                station = 'P'
            station_name = station + '{}{}'.format(p, d)
            local = LocalUpdate(args=args)
            local_g = local.independent_training(idp_G, idp_D, dataset, station_name, save_path)

        # 清除之前独立学习的主循环所占用的显存空间
        torch.cuda.empty_cache()

    # 写入训练过程数据
    fw_name = save_path + 'Fed_main_training_' + 'log.txt'
    fw_fed_main = open(fw_name, 'w+')
    fw_fed_main.write('iter\t G_loss\t D_loss\t G_train_MSE_loss\t G_test_RMSE_loss\t \n')

    # 联邦学习主循环
    with tqdm(range(args.epochs)) as tq:
        for iter in tq:  # 暂时取消self.args.local_ep *
            tq.set_description('Federated Updating')
            g_w_locals, g_loss_locals, d_w_locals, d_loss_locals = [], [], [], []
            usrs_weights = []
            local_g_mse_train_loss, local_g_rmse_test_loss = [], []
            # 用于随机抽取指定数量的参与者加入联邦学习
            # m = max(int(args.frac * args.num_users), 1)
            # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            idxs_users = range(len(stations))  # 手动选定参与者
            for idx in idxs_users:
                local = LocalUpdate(args=args, idx=idx)
                w_g, w_d, g_loss, d_loss, g_mse_train_loss, g_rmse_test_loss, train_no = local.train(
                    G=copy.deepcopy(G).to(args.device),
                    D=copy.deepcopy(D).to(args.device),
                    dataset=all_station_datasets[idx],
                )

                # 记录weights
                g_w_locals.append(copy.deepcopy(w_g))
                d_w_locals.append(copy.deepcopy(w_d))
                # 记录参与方的模型参数权重
                usrs_weights.append(train_no)
                # 记录loss
                g_loss_locals.append(g_loss)
                d_loss_locals.append(d_loss)
                # 记录G MSE lossr
                local_g_mse_train_loss.append(g_mse_train_loss)
                local_g_rmse_test_loss.append(g_rmse_test_loss)
            # 使用联邦学习算法更新全局权重
            if args.weights_avg:
                if args.wa_type == 'missing_ratio':
                    w = norm(args.missing_ratios, if_verse=True)
                elif args.wa_type == 'missing_number':
                    w = norm(usrs_weights)
                g_w_glob = FedWeightedAvg(g_w_locals, w, use_soft=True)
                d_w_glob = FedWeightedAvg(d_w_locals, w, use_soft=True)
            else:
                g_w_glob = FedAvg(g_w_locals)
                d_w_glob = FedAvg(d_w_locals)

            # 全局模型载入联邦平均化之后的模型参数
            G.load_state_dict(g_w_glob)
            D.load_state_dict(d_w_glob)

            # 学习率衰减
            if iter + 1 % args.d_lr_decay_step == 0:
                d_lr = d_lr * args.d_lr_decay
            if iter + 1 % args.fed_g_lr_decay_step == 0:
                g_lr = g_lr * args.fed_g_lr_decay

            # 打印训练过程的loss
            g_loss_avg = sum(g_loss_locals) / len(g_loss_locals)
            d_loss_avg = sum(d_loss_locals) / len(d_loss_locals)
            g_mse_train_loss_avg = sum(local_g_mse_train_loss) / len(local_g_mse_train_loss)
            g_rmse_test_loss_avg = sum(local_g_rmse_test_loss) / len(local_g_rmse_test_loss)

            g_mse_train_loss_avg_l.append(g_mse_train_loss_avg)
            g_mse_test_loss_avg_l.append(g_rmse_test_loss_avg)

            # print('Fed Main Loop Round {:3d}, Average G loss {:.3f}, Average D loss {:.3f}'.format(iter, g_loss_avg,
            #                                                                                        d_loss_avg))
            # print('Train_MSE: {:.4}'.format(g_mse_train_loss_avg.item()))
            # print('Test_RMSE: {:.4}'.format(g_rmse_test_loss_avg.item()))

            tq.set_postfix(Avg_G_loss=g_loss_avg.item(), Avg_D_loss=d_loss_avg.item(),
                           Fed_train_MSE=g_mse_train_loss_avg.item(),
                           Fed_test_RMSE=g_rmse_test_loss_avg.item())

            g_loss_train.append(g_loss_avg)
            d_loss_train.append(d_loss_avg)

            # 保存模型
            if iter % 5 == 0:
                # save_model_file = save_path.split('/')[2]
                file_name = save_path.split('/')
                save_model_file = file_name[2] + '/' + file_name[3]
                save_model(G, D, save_model_file)
            fw_fed_main.write('{}\t {:.5f}\t {:.5f}\t {:.5f}\t {:.5f}\t \n'.format(iter, g_loss_avg, d_loss_avg,
                                                                                   g_mse_train_loss_avg,
                                                                                   g_rmse_test_loss_avg))

    # 绘制曲线
    fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    loss_plot(axs[0, 0], g_loss_train, 'G train loss')
    loss_plot(axs[0, 1], d_loss_train, 'D train loss')
    loss_plot(axs[1, 0], g_mse_train_loss_avg_l, 'G MSE training loss')
    loss_plot(axs[1, 1], g_mse_test_loss_avg_l, 'RMSE on training dataset')

    plt.savefig(save_path + 'fed_{}.eps'.format(args.epochs))
    plt.savefig(save_path + 'fed_{}.png'.format(args.epochs))

    # 关闭写入
    fw_fed_main.close()
    plt.cla()  # 清除之前绘图
    plt.close()
    # 清空GPU缓存
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = args_parser()

    # 做实验
    exp_total_time = 1
    cross_validation_sets = 5

    results_saved_file = 'Fed_wGAN_results'
    results_plot_file = 'Fed_wGAN_plot_results'

    indicator_list = ['rmse', 'd2', 'r2', 'all_rmse']

    # params_test_list = [0.9]
    # test_param_name = 'p_hint'

    # 训练模式，是训练一次还是根据不同的参数训练多次
    training_model = 'Many_time'  # Many_time / One_time

    if training_model == 'Many_time':
        # params_test_list = [25, 40]
        params_test_list = ['(A5_A10_A15)_nCO_532r_One_time',
                            '(A5_A15_A30)_nCO_532r_One_time',
                            '(A5_A20_A30)_nCO_532r_One_time']
        test_param_name = 'missing_rate'
        Dname_prefix = '(A{}_A{}_A{})_nCO_532r'
        Dname_suffix = 'One_time'
    elif training_model == 'One_time':
        params_test_list = [1]
        test_param_name = 'One_time'
        # dataset_number = 'one_mi((A5_B10_E15)_111)'
        dataset_name = '(A5_B20_E30)_nCO_532r_One_time'
        # dataset_name = '(1P10_2P20_3P30)_532r_One_time'

    for param in params_test_list:

        print('**  {} params test: {}  **'.format(test_param_name, param))
        if training_model == 'Many_time':
            # dataset_name = Dname_prefix.format(param, param, param) + '_' + Dname_suffix
            dataset_name = param
        # dataset_name = 'one_mi_v1((A{})_1r_v3)'.format(param)
        exp_name = 'weightedAvg_{}_FedWGAI_T2'.format(dataset_name)
        ex_params_settings = {
            'algo_name': 'FedWA',
            'WeightAvg_method': args.weights_avg,
            'dataset': dataset_name,
            'datasets_number': cross_validation_sets,
            'epochs': args.epochs,
            'local_ep': args.local_ep,
            'local_bacthsize': args.local_bs,
            'n_critic': args.n_critic,
            'noise_input_dim': args.input_dim,
            'G_hidden_n': args.G_hidden_dim,
            'D_hidden_n': args.D_hidden_dim,
            'activate_function': 'LeakReLU',
            'optimizer': 'Adam',
            'fed_d_lr': args.fed_d_lr,
            'fed_g_lr': args.fed_g_lr,
            'idpt_d_lr': args.d_lr,
            'idpt_g_lr': args.g_lr,
            'lr_decay': args.g_lr_decay,
            'decay_step': args.g_lr_decay_step,
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
            print('\033[1;32m ============= Start training at datasets {} ==============\033[0m'.format(i))
            # 用于统计各种指标，建立相对应的文件夹
            result_save_file = './{}/'.format(results_saved_file) + exp_name + '/datasets_{}/'.format(i)

            for index in indicator_list:
                for model_name in ['fed', 'idpt']:
                    test_result_save_path = result_save_file + index + '/' + model_name
                    mkdir(test_result_save_path)

            for exp_t in range(exp_total_time):
                # 当前数据集
                args.dataset_path = './constructed_datasets/{}/{}/'.format(dataset_name, i)
                save_path_pre = result_save_file + str(exp_t) + '/'
                mkdir(save_path_pre)
                fed_main(args, save_path_pre)
                fed_gain_test_exp(args, result_save_file, exp_t)

        # 清空GPU缓存
        torch.cuda.empty_cache()

        print('>>> [Main Fed Process] Finished current training & testing!')

        result_save_root = './{}/'.format(results_saved_file) + exp_name + '/'
        plots_save_root = './{}/'.format(results_plot_file) + exp_name + '/'
        # indicator_name = 'all_rmse'
        indicator_list = ['rmse', 'd2', 'r2', 'all_rmse']
        leg = ['Federated', 'Independent']

        for indicator_name in indicator_list:
            # 建立保存结果的文件夹
            indicator_avg_results_csv_save_fpth = result_save_root + 'avg_' + indicator_name + '/'
            for mode in leg:
                mkdir(indicator_avg_results_csv_save_fpth + mode + '/')

            # 计算每个数据集的几次实验的均值
            for c in range(cross_validation_sets):
                results_logdir = [
                    result_save_root + 'datasets_' + str(c) + '/' + indicator_name + '/' + model_name + '/'
                    for model_name in ['fed', 'idpt']]

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
            plot_indicator_avg_results_m(results_logdir, fig_save_path, 'station', indicator_name, csv_save_fpth,
                                         select_dim=args.select_dim)
            # plot_indicator_results(results_logdir, fig_save_path, indicator_name)

            print("\033[0;34;40m >>>[Visulize] Finished save {} resluts figures! \033[0m".format(indicator_name))

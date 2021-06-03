import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from copy import deepcopy
import torch
from scipy.stats import pearsonr

# from LoadandCreateDatasets import get_saved_datasets
from CreateAndLoadDatasets import get_saved_datasets
from util_tools import get_time_stamp, save_as_csv, load_model, save2csv, mkdir, save_all_avg_results
from util_tools import sample_Z
from plot_indexes_resluts import plot_indicator_results, get_all_datasets
from param_options import args_parser

plt.style.use('seaborn')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def fed_test(G, g_input_dim, dataset, station, selected_dim=None):

    # 写入测试过程数据
    # fw_name = './results/fed_test_on_' + station + '_' + get_time_stamp() + '_log.txt'
    # fw_test = open(fw_name, 'w+')

    Dim = dataset['d']
    testX = dataset['test_x']
    testM = dataset['test_m']
    Test_No = dataset['test_no']
    min_val = dataset['min_val']
    max_val = dataset['max_val']

    # 对算法进行测试
    Z_mb = sample_Z(Test_No, g_input_dim)
    M_mb = testM
    X_mb = testX

    # New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce
    if g_input_dim == Dim:
        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce
    else:
        New_X_mb = Z_mb

    X_mb = torch.tensor(X_mb, device='cuda', dtype=torch.float32)
    M_mb = torch.tensor(M_mb, device='cuda', dtype=torch.float32)
    New_X_mb = torch.tensor(New_X_mb, device='cuda', dtype=torch.float32)

    with torch.no_grad():
        G_sample = G(New_X_mb, M_mb)
    '''
    # %% MSE Performance metric
    MSE_final = torch.mean(((1 - M_mb) * X_mb - (1 - M_mb) * G_sample) ** 2) / torch.mean(1 - M_mb)
    print('*** Model Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))
    '''

    # 修补好的数据 并转为numpy数组
    imputed_data = M_mb * X_mb + (1 - M_mb) * G_sample
    imputed_data = imputed_data.cpu().detach().numpy()
    # 原始数据，并转为numpy数组
    true_data = X_mb
    true_data = true_data.cpu().detach().numpy()

    # imputations data
    P = (1 - M_mb) * G_sample
    P = P.cpu().detach().numpy()
    # observed data
    O = (1 - M_mb) * X_mb
    O = O.cpu().detach().numpy()

    # 将mask也转为array数组
    test_mask = M_mb.cpu().detach().numpy()

    rmse_l = []
    d2_l = []
    r2_l = []

    all_mse = torch.mean(((1 - M_mb) * X_mb - (1 - M_mb) * G_sample) ** 2) / torch.mean(1 - M_mb)
    all_mse = all_mse.cpu().detach().numpy()
    all_rmse = np.sqrt(all_mse)

    for i in range(imputed_data.shape[1]):
        m = 1 - test_mask[:, i]     # imputations mask 缺失数据的mask
        N = m.sum()
        x_ = O[:, i].sum() / N
        x_pred_ = P[:, i].sum() / N
        mse = np.sum((O[:, i] - P[:, i]) ** 2)
        # mse = mean_squared_error(true_data[:, i], imputed_data[:, i])
        rmse = np.sqrt(mse / N)
        # d^2 评价指标
        d2 = 1 - ((O[:, i] - P[:, i]) ** 2).sum() / ((abs((O[:, i] - x_)*m) + abs((P[:, i] - x_)*m)) ** 2).sum()
        # R^2 评价指标
        r, p = pearsonr(true_data[:, i], imputed_data[:, i])
        r2 = r ** 2
        # 添加到用于记录的列表当中
        rmse_l.append(rmse)
        d2_l.append(d2)
        r2_l.append(r2)

        print('[{} Eval] {} RMSE: {}, D2: {}  scipy pearsonr R2: {}'.format(station, selected_dim[i], rmse, d2, r2))

    # print("Imputed test data:")
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})
    # print(imputed_data.cpu().detach().numpy())

    # fw_test.write('Fed glob model MSE Test on station {}: {}'.format(station, MSE_final))
    return rmse_l, d2_l, r2_l, all_rmse


def fed_gain_test_exp(args, save_path='', exp_num=None):

    if args.gan_categories == 'wGAN':
        from WAGIN_model import Generator, Discriminator
    else:
        from GAIN_model import Generator, Discriminator

    # parse args
    # args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    data_dim = len(args.select_dim)  # 真实数据维度，即生成器输出
    input_dim = args.input_dim  # 输入随机向量维度
    G_H_Dim = args.G_hidden_dim  # 设置G隐藏层的网络单元数量
    D_H_Dim = args.G_hidden_dim  # 设置D隐藏层的网络单元数量

    # 选择的站点sss
    stations = args.selected_stations
    # 针对load_dataset_v2，对不同参与方，每个参与方都有自己的站点数据添加
    if type(stations) == list:
        s_name = []
        for si in range(len(stations)):
            s_name.append('P'+str(si))
        stations = s_name

    # 载入数据并进行构造
    # 载入所有选定站点的数据集
    all_station_datasets = get_saved_datasets(args)

    # 建立GAIN model
    # 新建生成器网络G
    G = Generator(input_dim, G_H_Dim, data_dim).to(args.device)
    # 新建判别器网络D
    D = Discriminator(data_dim, D_H_Dim).to(args.device)

    # 模型存储文件夹
    # model_file = save_path.split('/')[2]
    model_file = save_path.split('/')
    model_file = model_file[2] + '/' + model_file[3]

    if args.independent_usrs_training:
        # 建立每个客户的独立个体
        independent_usrs_G = []
        independent_usrs_D = []

        for i, s in enumerate(args.selected_stations):
            # 为每个站点新建一个独立的个体网络,载入本地模型
            independent_G = deepcopy(G)
            independent_D = deepcopy(D)
            # 载入本地的各站点的独立模型
            if type(s) == list:
                s = 'P'
            station_name = s + '{}{}'.format(i, args.num_d[i])
            load_model(independent_G, independent_D, 'idpt', station_name, model_file)
            independent_usrs_G.append(independent_G)
            independent_usrs_D.append(independent_D)

        # 各站点的独立模型，testing 在测试集进行评估
        print('============== Independent station Test =======================')
        idpt_mse_on_test_results = []
        idpt_d2_on_test_results = []
        idpt_r2_on_test_results = []
        idpt_all_rmse_on_test_results = []
        for idpt_g, dataset, station in zip(independent_usrs_G, all_station_datasets, stations):
            mse, d2, p2, all_rmse = fed_test(idpt_g, input_dim, dataset, station, args.select_dim)
            idpt_mse_on_test_results.append(mse)
            idpt_d2_on_test_results.append(d2)
            idpt_r2_on_test_results.append(p2)
            idpt_all_rmse_on_test_results.append(all_rmse)
            print('[Idpt eval]- Station {} complete independent evaluation!\n'.format(station))
        idpt_save_csv_pt = save_path + 'rmse/idpt/idpt_mse_test_results_' + str(exp_num) + '.csv'
        # save_as_csv(idpt_save_csv_pt, idpt_mse_on_test_results, stations, 'MSE')
        save2csv(idpt_save_csv_pt, idpt_mse_on_test_results, stations, args.select_dim)
        idpt_save_csv_pt = save_path + 'd2/idpt/idpt_d2_test_results_' + str(exp_num) + '.csv'
        # save_as_csv(idpt_save_csv_pt, idpt_d2_on_test_results, stations, 'D2')
        save2csv(idpt_save_csv_pt, idpt_d2_on_test_results, stations, args.select_dim)
        idpt_save_csv_pt = save_path + 'r2/idpt/idpt_r2_test_results_' + str(exp_num) + '.csv'
        # save_as_csv(idpt_save_csv_pt, idpt_r2_on_test_results, stations, 'R2')
        save2csv(idpt_save_csv_pt, idpt_r2_on_test_results, stations, args.select_dim)
        # 保存all-rsme
        fed_save_csv_pt = save_path + 'all_rmse/idpt/idpt_all_rmse_test_results_' + str(exp_num) + '.csv'
        save_as_csv(fed_save_csv_pt, idpt_all_rmse_on_test_results, 'all_rmse')
        print('[Idpt eval]>>> Finished Idpt model evaluate on Test datasets!\n')

    # 载入本地联邦模型参数
    load_model(G, D, 'fed', station='', save_file=model_file)

    # testing 在测试集进行评估
    print('[Fed eval]===================== Federated station Test =======================')
    fed_mse_on_test_results = []
    fed_d2_on_test_results = []
    fed_r2_on_test_results = []
    all_rmse_on_test_results = []
    for dataset, station in zip(all_station_datasets, stations):
        mse, d2, p2, all_rmse = fed_test(G, input_dim, dataset, station, args.select_dim)
        fed_mse_on_test_results.append(mse)
        fed_d2_on_test_results.append(d2)
        fed_r2_on_test_results.append(p2)
        all_rmse_on_test_results.append(all_rmse)
        print('[Fed Eval] Station {} complete federated evaluation!\n'.format(station))
    # 保存到本地
    fed_save_csv_pt = save_path + 'rmse/fed/fed_mse_test_results_' + str(exp_num) + '.csv'
    save2csv(fed_save_csv_pt, fed_mse_on_test_results, stations, args.select_dim)
    # save_as_csv(fed_save_csv_pt, fed_mse_on_test_results, stations, 'MSE')
    fed_save_csv_pt = save_path + 'd2/fed/fed_d2_test_results_' + str(exp_num) + '.csv'
    # save_as_csv(fed_save_csv_pt, fed_d2_on_test_results, stations, 'D2')
    save2csv(fed_save_csv_pt, fed_d2_on_test_results, stations, args.select_dim)
    fed_save_csv_pt = save_path + 'r2/fed/fed_r2_test_results_' + str(exp_num) + '.csv'
    # save_as_csv(fed_save_csv_pt, fed_r2_on_test_results, stations, 'R2')
    save2csv(fed_save_csv_pt, fed_r2_on_test_results, stations, args.select_dim)
    # 保存all-rsme
    fed_save_csv_pt = save_path + 'all_rmse/fed/fed_all_rmse_test_results_' + str(exp_num) + '.csv'
    save_as_csv(fed_save_csv_pt, all_rmse_on_test_results, 'all_rmse')
    print('>>>[Fed Eval] Finished Fed model evaluate on Test datasets!')


def compute_indicator_results(args, indicator_list, results_file_saved_path, fig_save_file='', leg=None,
                              algorithm='Fed'):
    data = None
    if leg is None:
        leg = ['Fed', 'Independent']
    # 用于图像保存
    if fig_save_file == '':
        fig_save_fpath = results_file_saved_path + 'All_indicator_avg_results/'
    else:
        fig_save_fpath = fig_save_file + 'All_indicator_avg_results/'
    csv_save_fpath = results_file_saved_path + 'All_indicator_avg_results/'
    mkdir(fig_save_fpath)  # 新建文件夹用于保存图片
    mkdir(csv_save_fpath)  # 新建文件夹用于保存数据
    # 绘制测试结果图像
    print('====================== Save every component indicator result ========================')
    for indicator in indicator_list:
        print('{} results'.format(indicator))
        if indicator != 'all_rmse':
            if algorithm == 'Fed':
                _results_logdir = [results_file_saved_path + indicator + '/' + model_name + '/' for model_name in
                                   ['fed', 'idpt']]
            elif algorithm == 'EM':
                _results_logdir = [results_file_saved_path + indicator + '/']

            datas = get_all_datasets(_results_logdir, leg)

            if isinstance(datas, list):
                data = pd.concat(datas)

            unit_sets = data['Unit'].values.tolist()
            unit_sets = set(unit_sets)

            indicator_avg_list = []
            for mode in leg:
                avg_t = 0
                for u in unit_sets:
                    fed_avg_data = data[data.Condition == mode]
                    fed_avg_data = fed_avg_data[fed_avg_data.Unit == u][args.select_dim].values
                    avg_t += fed_avg_data
                indicator_avg = avg_t / len(unit_sets)
                indicator_avg_list.append(indicator_avg)

                # 保存到本地
                fed_save_csv_pt = csv_save_fpath + mode + '_' + indicator + '_avg_resluts.csv'
                save_all_avg_results(fed_save_csv_pt, indicator_avg, args.select_dim, args.selected_stations)
                print('***** ' + mode + ' avg: ', indicator_avg)

            c = len(args.select_dim)
            r = len(args.selected_stations)
            x = [h for h in range(r)]
            # plot style
            fig, axs = plt.subplots(2, 3, constrained_layout=True)

            for i in range(c):
                if algorithm == 'Fed':
                    axs[i // 3, i % 3].plot(x, indicator_avg_list[0][:, i], label='Federated')  # Fed
                    axs[i // 3, i % 3].plot(x, indicator_avg_list[1][:, i], label='Independent')  # Local
                else:
                    axs[i // 3, i % 3].plot(x, indicator_avg_list[0][:, i], label='EM')
                axs[i // 3, i % 3].set_xlabel('Station')
                axs[i // 3, i % 3].set_ylabel(indicator)
                axs[i // 3, i % 3].set_title(args.select_dim[i])
                # if indicator == 'rmse':
                #     axs[i // 3, i % 3].set_ylim(0, 0.1)
                # else:
                #     axs[i // 3, i % 3].set_ylim(0, 1.0)
                # axs[i // 3, i % 3].legend(loc='upper right', fontsize=8)
            plt.legend(loc='upper right', fontsize=8)

            save_path = fig_save_fpath + indicator + '_avg_resluts.svg'
            plt.savefig(save_path)
            # plot_indicator_results(results_logdir, fig_save_path, component)

            print('')
        elif indicator == 'all_rmse':
            if algorithm == 'EM':
                results_logdir = [results_file_saved_path + 'all_rmse/']
                plot_indicator_results(results_logdir, fig_save_fpath, 'all_rmse', xaxis='station', leg=['EM'])

        plt.clf()
        plt.close()
    return True


def plot_all_algorithm_indicator_results(args, indicator_list, results_file_saved_path, fig_save_file='', leg=None):
    """
    用于绘制所有算法的指标结果图
    :param args:
    :param indicator_list:
    :param results_file_saved_path:
    :param fig_save_file:
    :param leg:
    :return:
    """
    data = None
    if leg is None:
        leg = ['Fed', 'Independent']
    # 用于图像保存
    if fig_save_file == '':
        fig_save_fpath = results_file_saved_path + 'All_indicator_avg_results/'
    else:
        fig_save_fpath = fig_save_file + 'All_indicator_avg_results/'
    csv_save_fpath = results_file_saved_path + 'All_indicator_avg_results/'
    mkdir(fig_save_fpath)  # 新建文件夹用于保存图片
    mkdir(csv_save_fpath)  # 新建文件夹用于保存数据
    # 绘制测试结果图像
    print('====================== Save every component indicator result ========================')
    for indicator in indicator_list:
        print('{} results'.format(indicator))
        if indicator != 'all_rmse':
            _results_logdir = [results_file_saved_path + indicator + '/' + model_name + '/' for model_name in
                               ['fed', 'idpt', 'em']]

            datas = get_all_datasets(_results_logdir, leg)

            if isinstance(datas, list):
                data = pd.concat(datas)

            unit_sets = data['Unit'].values.tolist()
            unit_sets = set(unit_sets)

            indicator_avg_list = []
            for mode in leg:
                avg_t = 0
                for u in unit_sets:
                    fed_avg_data = data[data.Condition == mode]
                    fed_avg_data = fed_avg_data[fed_avg_data.Unit == u][args.select_dim].values
                    avg_t += fed_avg_data
                indicator_avg = avg_t / len(unit_sets)
                indicator_avg_list.append(indicator_avg)

                # 保存到本地
                fed_save_csv_pt = csv_save_fpath + mode + '_' + indicator + '_avg_resluts.csv'
                save_all_avg_results(fed_save_csv_pt, indicator_avg, args.select_dim, args.selected_stations)
                print('***** ' + mode + ' avg: ', indicator_avg)

            c = len(args.select_dim)
            r = len(args.selected_stations)
            x = [h for h in range(r)]
            # plot style
            fig, axs = plt.subplots(2, 3, constrained_layout=True)

            for i in range(c):
                for j in range(len(indicator_avg_list)):
                    axs[i // 3, i % 3].plot(x, indicator_avg_list[j][:, i], label=leg[j])
                axs[i // 3, i % 3].set_xlabel('Station')
                axs[i // 3, i % 3].set_ylabel(indicator)
                axs[i // 3, i % 3].set_title(args.select_dim[i])
                # if indicator == 'rmse':
                #     axs[i // 3, i % 3].set_ylim(0, 0.1)
                # else:
                #     axs[i // 3, i % 3].set_ylim(0, 1.0)
                # axs[i // 3, i % 3].legend(loc='upper right', fontsize=8)
            plt.legend(loc='upper right', fontsize=8)

            save_path = fig_save_fpath + indicator + '_avg_resluts.svg'
            plt.savefig(save_path)
            # plot_indicator_results(results_logdir, fig_save_path, component)

            print('')
        elif indicator == 'all_rmse':
            # 计算all_rmse值，并保存到表格
            _results_logdir = [results_file_saved_path + 'all_rmse/']

            plot_indicator_results(_results_logdir, fig_save_fpath, 'all_rmse', xaxis='station', leg=leg)

        plt.clf()
        plt.close()
    return True


def plot_indicator_avg_results_m(logdir, save_path, xaxis, value, save_csv_fpth, leg=None, condition=None):
    # 重新实现了seaborn的带有均值线的多次实验结果图
    if leg is None:
        leg = ['Federated GAN', 'Local GAN']

    datas = get_all_datasets(logdir, leg)
    # 创建绘图
    fig, ax = plt.subplots()

    if isinstance(datas, list):
        data = pd.concat(datas)
        print('*** data: \n', data)

    unit_sets = data['Unit'].values.tolist()
    unit_sets = set(unit_sets)

    condition_sets = data['Condition'].values.tolist()
    condition_sets = set(condition_sets)

    xaxis_sets = data[xaxis].values.tolist()
    xaxis_sets = set(xaxis_sets)

    indicator_avg_list = []
    # 不同的condition
    for mode in condition_sets:
        avg_t = 0
        # 计算被标记的不同unit之间的均值
        condition_min_list = []
        condition_max_list = []
        condition_unit_data_list = []
        condition_data = data[data.Condition == mode]
        for u in unit_sets:
            condition_unit_data = condition_data[condition_data.Unit == u][value].values
            condition_unit_data_list.append(condition_unit_data.reshape((1, -1)))
            avg_t += condition_unit_data

        # 用于替代seaborn中的tsplot绘图函数
        def tsplot(ax, x, data, **kw):
            est = np.mean(data, axis=0)
            sd = np.std(data, axis=0)
            cis = (est - sd, est + sd)
            x = np.array(x).astype(dtype=np.str)
            ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
            ax.plot(x, est, label=mode, **kw)
            ax.tick_params(labelsize=13)
            ax.set_ylabel('RMSE', size=13)
            ax.set_xlabel('Participant', size=13)
            ax.legend()
            ax.margins(x=0)

            return est, sd

        # 将几次的实验数据进行拼接，形成一个（station， exp_time）形状的数组
        all_condition_unit_data = np.concatenate(condition_unit_data_list, axis=0)
        xaxis_from_sets = [i for i in xaxis_sets]

        indicator_avg, indicator_std = tsplot(ax, xaxis_from_sets, all_condition_unit_data)

        # for x in xaxis_sets:
        #     condition_xaxis_data = condition_data[condition_data[xaxis] == x][value].values
        #     condition_min = np.min(condition_xaxis_data)
        #     condition_max = np.max(condition_xaxis_data)
        #     condition_min_list.append(condition_min)
        #     condition_max_list.append(condition_max)
        # indicator_avg = avg_t / len(unit_sets)
        # indicator_avg_list.append(indicator_avg)
        #
        # y_est = indicator_avg
        # ax.plot(xaxis_from_sets, y_est, '-', label=mode)
        # ax.fill_between(xaxis_from_sets, condition_min_list, condition_max_list, alpha=0.2)
        # 保存到本地
        save_avg_csv_pt = save_csv_fpth + mode + '_' + value + '_avg_resluts.csv'
        mode_save_data = {'RMSE': indicator_avg, 'std': indicator_std}
        # 使用pandas保存成csv
        dataframe = pd.DataFrame(mode_save_data)
        dataframe.to_csv(save_avg_csv_pt, index=True, sep=',')

        # save_all_avg_results(fed_save_csv_pt, indicator_avg, [value], xaxis)
    save_path = save_path + value + '_results'
    plt.savefig(save_path + '.eps')
    plt.savefig(save_path + '.svg')

    plt.close()


def run_multi_cross_validation_datasets_test():
    """
    用于计算每个数据集的学习训练测试结果的总学习测试结果的平均
    :return:
    """

    args = args_parser()

    # 做实验
    exp_total_time = 10
    cross_validation_sets = 5
    exp_name = 'FedWeightAvg(soft)_6_dn(15)_100_32(0.001_0.001_bs_128)_32(phint_0.95)'

    indicator_list = ['rmse', 'd2', 'r2', 'all_rmse']

    # for i in range(cross_validation_sets):
    #     print('============= Start training at datasets {} =============='.format(i))
    #     # 用于统计各种指标，建立相对应的文件夹
    #     result_save_file = './results_v2/' + exp_name + '/datasets_{}/'.format(i)
    #     plots_file = './plot_results/' + exp_name + '/datasets_{}/'.format(i)
    #     for exp_t in range(exp_total_time):
    #         # 当前数据集
    #         args.dataset_path = './constructed_datasets_6_dn(5)/{}/'.format(i)
    #         fed_gain_test_exp(args, result_save_file, i)

    # print('>>> Finished training & testing!')
    result_save_root = './results_v3/' + exp_name + '/'
    plots_save_root = './plot_results/' + exp_name + '/'
    indicator_name = 'all_rmse'
    leg = ['Fed', 'Independent']
    '''
    # 建立保存结果的文件夹
    indicator_results_csv_save_fpth = result_save_root + indicator_name + '/'
    for mode in leg:
        mkdir(indicator_results_csv_save_fpth+mode+'/')

    # 计算每个数据集的几次实验的均值
    for c in range(cross_validation_sets):
        results_logdir = [result_save_root + 'datasets_' + str(c) + '/' + indicator_name + '/' + model_name + '/'
                          for model_name in ['fed', 'idpt']]

        compute_avg_of_data_in_file(args, c, results_logdir, indicator_results_csv_save_fpth,
                                    indicator_name)
    '''
    # 绘制测试结果图像
    print('====================== Save every component indicator result ========================')

    print('{} results'.format(indicator_name))

    results_logdir = [result_save_root + indicator_name + '/' + model_name + '/' for model_name in leg]
    fig_save_path = plots_save_root + indicator_name + '/'
    csv_save_fpth = result_save_root + indicator_name + '/'
    mkdir(fig_save_path)

    # 计算几次实验的最后的平均值并画出均值和方差曲线图
    plot_indicator_avg_results_m(results_logdir, fig_save_path, 'station', indicator_name, csv_save_fpth)
    # plot_indicator_results(results_logdir, fig_save_path, indicator_name)

    print(">>> Finished save resluts figures!")


if __name__ == '__main__':
    args = args_parser()

    # 做实验
    exp_total_time = 1
    cross_validation_sets = 1

    results_saved_file = 'Fed_wGAN_results'
    results_plot_file = 'Fed_wGAN_plot_results'

    indicator_list = ['rmse', 'd2', 'r2', 'all_rmse']

    # params_test_list = [0.9]
    # test_param_name = 'p_hint'

    # 训练模式，是训练一次还是根据不同的参数训练多次
    training_model = 'One_time'  # Many_time / One_time

    if training_model == 'Many_time':
        params_test_list = [5]
        test_param_name = 'missing_rate'
        Dname_prefix = 'one_mi_v1((A{})_1r)'
        Dname_suffix = ''
    elif training_model == 'One_time':
        params_test_list = [1]
        test_param_name = 'One_time'
        # dataset_number = 'one_mi((A5_B10_E15)_111)'
        dataset_name = '(A5_A10_A15)_nCO_532r_One_time'
        # dataset_name = '(1P10_2P20_3P30)_532r_One_time'

    for param in params_test_list:

        print('**  {} params test: {}  **'.format(test_param_name, param))
        if training_model == 'Many_time':
            dataset_name = Dname_prefix.format(param, param, param) + '_' + Dname_suffix
        # dataset_name = 'one_mi_v1((A{})_1r_v3)'.format(param)
        exp_name = 'C_Test_{}_FedWGAI_T1'.format(dataset_name)
        # 存储主文件路径
        result_save_main_file = './{}/'.format(results_saved_file) + exp_name + '/'
        mkdir(result_save_main_file)

        for i in range(cross_validation_sets):
            print('============= Start training at datasets {} =============='.format(i))
            # 用于统计各种指标，建立相对应的文件夹
            result_save_file = './{}/'.format(results_saved_file) + exp_name + '/datasets_{}/'.format(i)

            for index in indicator_list:
                for model_name in ['fed', 'idpt']:
                    test_result_save_path = result_save_file + index + '/' + model_name
                    mkdir(test_result_save_path)

            for exp_t in range(exp_total_time):
                # 当前数据集
                args.dataset_path = './constructed_datasets/{}/{}/'.format(dataset_name, i)

                print('[Main Fed Process]******* Training epoch {} *******'.format(exp_t))
                save_path_pre = result_save_file + str(exp_t) + '/'
                mkdir(save_path_pre)
                fed_gain_test_exp(args, result_save_file, exp_t)

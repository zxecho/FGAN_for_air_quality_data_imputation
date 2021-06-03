import matplotlib
from copy import deepcopy
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from GAIN_model import Generator, Discriminator, weights_init
from CreateAndLoadDatasets import get_saved_datasets, get_saved_datasets_vall
from util_tools import get_time_stamp, save_as_csv, load_model, save2csv
from util_tools import sample_Z, mkdir
from param_options import args_parser
from plot_indexes_resluts import plot_indicator_results, get_all_datasets
matplotlib.use('Agg')


def fed_test(G, g_input_dim, dataset, station, selected_dim=None):

    # 写入测试过程数据
    # fw_name = './results/fed_test_on_' + station + '_' + get_time_stamp() + '_log.txt'
    # fw_test = open(fw_name, 'w+')

    G.eval()

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

        print('{} RMSE: {}, D2: {}  scipy pearsonr R2: {}'.format(selected_dim[i], rmse, d2, r2))

    # print("Imputed test data:")
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})
    # print(imputed_data.cpu().detach().numpy())

    # fw_test.write('Fed glob model MSE Test on station {}: {}'.format(station, MSE_final))
    return rmse_l, d2_l, r2_l, all_rmse


def gain_test_exp(args, save_path='', exp_num=None):

    if args.gan_categories == 'wGAN':
        from WAGIN_model import Generator, Discriminator
    else:
        from GAIN_model import Generator, Discriminator

    # parse args
    # args = args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        for i, s in enumerate(args.selected_stations):
            # 为每个站点新建一个独立的个体网络,载入本地模型
            independent_G = deepcopy(G)
            independent_D = deepcopy(D)
            # 载入本地的各站点的独立模型
            station_name = s + '{}{}'.format(i, args.num_d[i])
            independent_G, _ = load_model(independent_G, independent_D, 'idpt', station_name, model_file)
            independent_usrs_G.append(independent_G)

        # 各站点的独立模型，testing 在测试集进行评估
        print('\033[0;31;40m[Eval] Independent station Test \033[0m')
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
            print('\033[0;32;40m [Eval] - Station {} complete independent evaluation! \033[0m\n'.format(station))
        idpt_save_csv_pt = save_path + 'rmse/idpt/idpt_rmse_test_results_' + str(exp_num) + '.csv'
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
        print('\033[0;33;40m >>> [Eval] Finished Idpt model evaluate on Test datasets! \033[0m\n')


def plot_indicator_avg_results_m(logdir, save_path, xaxis, value, save_csv_fpth, leg=None, condition=None):
    # 重新实现了seaborn的带有均值线的多次实验结果图
    if leg is None:
        leg = ['Fed', 'Local']

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
            ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
            ax.plot(x, est, label=mode, **kw)
            ax.legend()
            ax.margins(x=0)

            return est, sd

        # 将几次的实验数据进行拼接，形成一个（station， exp_time）形状的数组
        all_condition_unit_data = np.concatenate(condition_unit_data_list, axis=0)
        xaxis_from_sets = [i for i in xaxis_sets]

        indicator_avg, indicator_std = tsplot(ax, xaxis_from_sets, all_condition_unit_data)

        # 保存到本地
        save_avg_csv_pt = save_csv_fpth + mode + '_' + value + '_avg_resluts.csv'
        mode_save_data = {xaxis: indicator_avg, 'std': indicator_std}
        # 使用pandas保存成csv
        dataframe = pd.DataFrame(mode_save_data)
        dataframe.to_csv(save_avg_csv_pt, index=True, sep=',')

        # save_all_avg_results(fed_save_csv_pt, indicator_avg, [value], xaxis)
    save_path = save_path + value + '_results'
    plt.savefig(save_path + '.eps')
    plt.savefig(save_path + '.svg')

    plt.close()


if __name__ == '__main__':
    args = args_parser()

    # 做实验
    exp_total_time = 1
    cross_validation_sets = 1

    dataset_number = 'one_mi((A10)_1)'

    results_saved_file = 'results_one_dn'
    results_plot_file = 'plot_results_one_dn'

    indicator_list = ['rmse', 'd2', 'r2', 'all_rmse']
    model_name_list = ['idpt']

    params_test_list = [5]
    test_param_name = 'missing_ratio'

    # params_test_list = [0.5, 1.0, 10]
    # test_param_name = 'alpha'

    for param in params_test_list:

        print('**  {} params test: {}  **'.format(test_param_name, param))
        dataset_number = 'one_mi((A{})_1_vall)'.format(param)
        exp_name = 'GAIN_alpha1_{}_latest'.format(dataset_number)

        # 存储主文件路径
        result_save_main_file = './{}/'.format(results_saved_file) + exp_name + '/'

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
                gain_test_exp(args, result_save_file, exp_t)

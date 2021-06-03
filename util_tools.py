import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

plt.style.use('ggplot')
import os
from plot_indexes_resluts import get_all_datasets


DIV_LINE_WIDTH = 50
units = dict()


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


# ================= Compute various indictors =================
def compute_r2(x, y):
    r, p = pearsonr(x, y)
    r2 = r ** 2
    return r2


def compute_d2(x, y, m):
    m = 1 - m  # imputations mask 缺失数据的mask
    N = m.sum()
    x_ = (x*m).sum() / N

    d2 = 1 - ((x - y) ** 2).sum() / ((abs((x - x_)*m) + abs((y - x_)*m)) ** 2).sum()

    return d2


def compute_rmse(x, y, m):
    m = 1 - m  # imputations mask 缺失数据的mask
    N = m.sum()
    mse = np.sum((x - y) ** 2)
    rmse = np.sqrt(mse / N)

    return rmse


def compute_all_rmse(x, y, m):
    mse = np.mean(((1 - m) * x - (1 - m) * y) ** 2) / np.mean(1 - m)
    rmse = np.sqrt(mse)

    return rmse


def norm(x, if_verse=False):
    x = np.array(x)
    if if_verse:
        x = 1 - x
    x = x / x.sum()
    return x.tolist()


# =========================== Ploting ===================
# 用于计算不同文件夹下的实验指标结果的均值
def compute_avg_of_data_in_file(args, dataset_index, results_logdir, csv_save_fpath, indicator_name, leg=None):
    if leg is None:
        leg = ['Fed', 'Independent']

    datas = get_all_datasets(results_logdir, leg)

    if isinstance(datas, list):
        data = pd.concat(datas)

    unit_sets = data['Unit'].values.tolist()
    unit_sets = set(unit_sets)

    # 针对load_dataset_v2，对不同参与方，每个参与方都有自己的站点数据添加
    # if type(args.selected_stations[0]) == list:
    #     station_names = args.clients
    # else:
    #     station_names = args.selected_stations

    station_names = args.clients

    indicator_avg_list = []
    for mode in leg:
        avg_t = 0
        for u in unit_sets:
            fed_avg_data = data[data.Condition == mode]
            if indicator_name == 'all_rmse':
                fed_avg_data = fed_avg_data[fed_avg_data.Unit == u][indicator_name].values
            else:
                fed_avg_data = fed_avg_data[fed_avg_data.Unit == u][args.select_dim].values
            avg_t += fed_avg_data
        indicator_avg = avg_t / len(unit_sets)
        indicator_avg_list.append(indicator_avg)

        # 保存到本地
        fed_save_csv_pt = csv_save_fpath + mode + '/' + 'dataset_' + \
                          str(dataset_index) + '_' + mode + '_' + indicator_name + '_avg_resluts.csv'
        if indicator_name == 'all_rmse':
            save_all_avg_results(fed_save_csv_pt, indicator_avg, [indicator_name], station_names)
        else:
            save_all_avg_results(fed_save_csv_pt, indicator_avg, args.select_dim, station_names)
        print('[C] ' + mode + ' avg: ', indicator_avg)


def loss_plot(axs, loss_data, name=None):
    axs.plot(range(len(loss_data)), loss_data, label=name)
    axs.tick_params(labelsize=13)
    axs.set_ylabel(name, size=13)
    axs.set_xlabel('Epochs', size=13)
    axs.legend()
    # axs.set_title(name)


# =========== Other tool functions ====================
def normalization(data, parameters=None):
    """Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    """

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

            # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    """
    Renormalize data from [0, 1] range to the original range.

    Args:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
      - renorm_data: renormalized original data
    """

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data


def rounding(imputed_data, data_x):
    """
    Round imputed data for categorical variables.
    对修补的数据进行四舍五入
    Args:
      - imputed_data: imputed data
      - data_x: original data with missing values

    Returns:
      - rounded_data: rounded imputed data
    """

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


def get_time_stamp():
    # datetime获取当前时间，数组格式
    now = datetime.datetime.now()
    stamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    # print(stamp)
    return stamp


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False


def sample_idx(m, n):
    """
    :arg
    n: 采样index的个数
    m: 样本总个数
    """
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


def sample_Z(m, n, scale_factor=0.1):
    """
    用于采样随机扰动
    :param scale_factor:
    :param m: batch_size
    :param n: Dim, 数据属性维度
    :return:
    """
    return np.random.uniform(0., 1, size=[m, n])
    # return np.random.normal(0.5, 0.2, size=[m, n]) * scale_factor


def save_model(G, D, save_file=''):
    mkdir('./saved_model/' + save_file + '/')
    torch.save(D.state_dict(), './saved_model/' + save_file + '/D.pkl')
    torch.save(G.state_dict(), './saved_model/' + save_file + '/G.pkl')


def load_model(G, D, tag='', station='', save_file=''):
    g_load_path = ''
    d_load_path = ''
    if tag == 'fed':
        g_load_path = './saved_model/' + save_file + '/G.pkl'
        d_load_path = './saved_model/' + save_file + '/D.pkl'
    elif tag == 'idpt':
        g_load_path = './saved_model/' + save_file + '/independent_' + station + '_G.pkl'
        d_load_path = './saved_model/' + save_file + '/independent_' + station + '_D.pkl'

    G.load_state_dict(torch.load(g_load_path))
    D.load_state_dict(torch.load(d_load_path))

    return G, D


def save_as_csv(fpt, data, key='MSE'):
    dataframe = pd.DataFrame({key: data})
    dataframe.to_csv(fpt, index=False, sep=',')


def save2csv(fpt, data, columns, index):
    if len(data) != len(columns):
        data = [data]
    data = {key: value for key, value in zip(columns, data)}
    print('*** data: \n', data)
    dataframe = pd.DataFrame(data, columns=columns, index=index)
    # 转置
    dataframe = pd.DataFrame(dataframe.values.T, index=dataframe.columns, columns=dataframe.index)
    dataframe.to_csv(fpt, index=True, sep=',')


def save_all_avg_results(fpt, data, columns, index):
    if len(columns) == len(data):
        data = {key: value for key, value in zip(columns, data)}
    elif len(columns) == 1:
        data = {columns[0]: data}
    print('*** data: \n', data)
    dataframe = pd.DataFrame(data, columns=columns, index=index)
    dataframe.to_csv(fpt, index=True, sep=',')


def save2json(fpt, data, colum_names):
    with open(fpt, 'w+') as f:
        json.dump({key: value for key, value in zip(colum_names, data)}, f)


def load_json(fpt):
    with open(fpt) as f:
        json_data = json.load(f)
    return json_data


if __name__ == '__main__':
    pass

    # 用于绘制几次平均结果
    # logdir = ["E:\\zx\\Fed-AQ\\training_results\\Fed\\", "E:\\zx\\Fed-AQ\\training_results\\Independent\\"]
    # leg = ['Fed', 'Independent']
    #
    # datas = get_all_datasets(logdir, leg)
    # print(datas)
    # xaxis = 'station'
    # value = 'MSE'
    # condition = 'Condition'
    # smooth = 1
    # estimator = getattr(np, 'mean')
    #
    # plot_data(datas, xaxis=xaxis, value=value, condition=condition, estimator=estimator)
    # plt.show()


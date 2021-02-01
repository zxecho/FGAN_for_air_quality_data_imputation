import numpy as np
import random
import pandas as pd
from util_tools import save2json, load_json, mkdir, normalization
import matplotlib.pyplot as plt
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 5. Train Rate
train_rate = 0.8


# 载入数据
def load_datasets(AQS, select_dim, return_pd=False, **kwargs):
    # 载入气象站数据
    # 载入数据处理
    dataset_list = []
    pd_dataset_list = []
    q = 0  # 用于标记季节
    si = 0  # 用于采样间隔（h）
    for station in AQS:
        station_data_list = []
        pd_data_list = []
        # 改变日期格式，如果有规定要载入按季节的数据q
        if 'quarter_index' in kwargs.keys():
            quarter_index = kwargs['quarter_index'][q]
            q += 1

        if 'sample_interval' in kwargs.keys():
            interval = kwargs['sample_interval'][si]
            si += 1

        for i in range(1, 3):
            data_file = station + '_' + str(i) + '.xlsx'
            print(data_file)
            data_file_path = './air_quality_datasets/' + data_file
            air_data_df = pd.read_excel(data_file_path)

            # 改变日期格式，如果有规定要载入按季节的数据
            if 'quarter_index' in kwargs.keys():
                air_data_df.set_index(pd.to_datetime(air_data_df["Date"]), inplace=True)

            # 载入不同采样间隔的数据
            if 'sample_interval' in kwargs.keys():
                air_data_df = air_data_df[air_data_df.index % interval == 0]

            df_sel = air_data_df.loc[:, select_dim]

            df_noNaN = df_sel[df_sel.notnull().sum(axis=1) == len(select_dim)]

            Data = df_noNaN.values
            print('Data %d shape : \n' % i, Data.shape)
            station_data_list.append(Data)
            pd_data_list.append(df_noNaN)
        station_data = np.concatenate((station_data_list[0], station_data_list[1]), axis=0)
        pd_data = pd.concat(pd_data_list, axis=0)

        if 'quarter_index' in kwargs.keys():
            quarter_data_list = []
            if quarter_index != [1, 2, 3, 4]:       # 如果需要提取个别季节数据的才提取他则不用
                for idx in quarter_index:
                    station_data = pd_data.loc[pd_data.index.quarter == idx]
                    quarter_data_list.append(station_data)
                quarter_data = pd.concat(quarter_data_list, axis=0)
                dataset_list.append(quarter_data.values)
            else:
                dataset_list.append(pd_data.values)
        else:
            dataset_list.append(station_data)

        print('station_data shape: ', station_data.shape)
        pd_data_list.append(pd_data)

    if return_pd:
        return pd_data_list
    else:
        return dataset_list


# 构造缺失mask（包含行，列，散点）
def construct_missing_mask(data, missing_rate=0.2):
    L = data.shape[0]
    dim = data.shape[1]
    data_shape = data.shape

    line_miss_num = int(1 / 1000 * L)
    row_miss_num = int(1 / 1000 * L)
    # 缺失率
    p_miss = missing_rate

    # 行缺失
    while True:
        rand_m1 = np.random.randint(0, L, line_miss_num)
        u = np.unique(rand_m1)
        if len(u) == line_miss_num:
            break
    line_missing_p_num = line_miss_num * dim  # 计算出行缺失的数量

    # 列缺失
    rand_m2_v = np.random.randint(0, dim, row_miss_num)
    rand_m2_l = np.random.randint(1, L, row_miss_num)
    # 每个列缺失的数据个数，在一定范围内随机
    missing_len = np.random.randint(2, 6, row_miss_num)
    # 计算出列缺失的数量
    row_missing_p_num = 0
    for i in range(len(missing_len)):
        row_missing_p_num += missing_len[i]

    # 点缺失
    miss_num = p_miss * L * dim
    miss_p_num = miss_num - row_missing_p_num - line_missing_p_num
    prob_missing_p = round(miss_p_num / (L * dim), 3)
    # 构造missing 矩阵
    Missing = np.zeros(data_shape)
    p_miss_vec = prob_missing_p * np.ones((L, 1))
    for i in range(dim):
        A = np.random.uniform(0., 1., size=[L, ])
        B = A > p_miss_vec[i]
        Missing[:, i] = 1. * B

    Missing[rand_m1, :] = Missing[rand_m1, :] * 0  # 行缺失赋值

    # 列缺失赋值
    for i in range(row_miss_num):
        Missing[rand_m2_l[i]: rand_m2_l[i] + missing_len[i], rand_m2_v[i]] = Missing[rand_m2_l[i]: rand_m2_l[i] + missing_len[i],
                                                                             rand_m2_v[i]] * 0
    # print('Missing: \n', Missing)
    real_missing_num = 1 - Missing
    real_missing_num = real_missing_num.sum()
    real_missing_rate = real_missing_num / (L * dim)
    print('real missing rate : ', real_missing_rate, 'expect missing rate: ', p_miss)

    return Missing


# 构造缺失mask（包含行，散点）
def construct_missing_mask_v2(data, missing_rate=0.2):
    L = data.shape[0]
    dim = data.shape[1]
    data_shape = data.shape

    line_miss_num = int(1 / 1000 * L)
    row_miss_num = int(1 / 1000 * L)
    # 缺失率
    p_miss = missing_rate

    # 行缺失
    line_missing_p_num = 0      # 行缺失数总量
    while True:
        rand_m1 = np.random.randint(0, L, line_miss_num)
        u = np.unique(rand_m1)
        if len(u) == line_miss_num:
            break

    lm_idx_list = []    # 用于记录随机采取的行缺失index
    for i in range(line_miss_num):
        # 行缺失的维度index
        rand_m1_l = random.sample(range(0, dim+1), 2)
        lm_idx_list.append(rand_m1_l)
        low = min(rand_m1_l)
        high = max(rand_m1_l)
        line_missing_p_num += (high - low)

    # 列缺失
    rand_m2_v = np.random.randint(0, dim, row_miss_num)
    rand_m2_l = np.random.randint(1, L, row_miss_num)
    # 每个列缺失的数据个数，在一定范围内随机
    missing_len = np.random.randint(2, 6, row_miss_num)
    # 计算出列缺失的数量
    row_missing_p_num = np.array(missing_len).sum()

    # 点缺失
    miss_num = p_miss * L * dim
    miss_p_num = miss_num - line_missing_p_num - row_missing_p_num
    prob_missing_p = round(miss_p_num / (L * dim), 3)
    # 构造missing 矩阵
    Missing = np.zeros(data_shape)
    p_miss_vec = prob_missing_p * np.ones((L, 1))
    for i in range(dim):
        A = np.random.uniform(0., 1., size=[L, ])
        B = A > p_miss_vec[i]
        Missing[:, i] = 1. * B

    # 行缺失赋值
    for l, rand_m1_l in zip(rand_m1, lm_idx_list):
        low = min(rand_m1_l)
        high = max(rand_m1_l)
        Missing[l, low:high] = Missing[l, low:high] * 0

    # 列缺失赋值
    for i in range(row_miss_num):
        if rand_m2_l[i] + missing_len[i] < Missing.shape[0]:
            Missing[rand_m2_l[i]: rand_m2_l[i] + missing_len[i], rand_m2_v[i]] = Missing[rand_m2_l[i]: rand_m2_l[i] + missing_len[i],
                                                                                 rand_m2_v[i]] * 0
        else:
            Missing[rand_m2_l[i]:, rand_m2_v[i]] = Missing[rand_m2_l[i]:, rand_m2_v[i]] * 0

    # print('Missing: \n', Missing)
    real_missing_num = 1 - Missing
    real_missing_num = real_missing_num.sum()
    real_missing_rate = real_missing_num / (L * dim)
    print('real missing rate : ', real_missing_rate, 'expect missing rate: ', p_miss)

    return Missing


# 构造缺失数据
def construct_train_test_dataset(Data, missing_rate=0.2, save=False,
                                 dataset_path='', station='', random=True, one_station=False, **kwargs):
    # 函数参数
    """
    :param missing_rate:
    :param dataset_path:
    :param Data: 构造的数据
    :param save: 是否保存
    :param station: 站点名称
    :return:
    """
    # 选择制定数量数据
    No = Data.shape[0]

    # 数据参数
    Dim = Data.shape[1]  # 数据维度

    # Normalization (0 to 1)
    Min_Val = np.zeros(Dim)
    Max_Val = np.zeros(Dim)

    for i in range(Dim):
        Min_Val[i] = np.min(Data[:, i])
        Data[:, i] = Data[:, i] - np.min(Data[:, i])
        Max_Val[i] = np.max(Data[:, i])
        Data[:, i] = Data[:, i] / (np.max(Data[:, i]) + 1e-6)

    # Missing introducing
    # p_miss_vec = p_miss * np.ones((Dim, 1))
    #
    # Missing = np.zeros((No, Dim))
    #
    # for i in range(Dim):
    #     A = np.random.uniform(0., 1., size=[len(Data), ])
    #     B = A > p_miss_vec[i]
    #     Missing[:, i] = 1. * B

    # 分离出训练和验证的数据集

    if random:
        idx = np.random.permutation(No)
    else:
        idx = [i for i in range(No)]

    Train_No = int(No * train_rate)
    Test_No = No - Train_No

    # Train / Test Features
    trainX = Data[idx[:Train_No], :]
    testX = Data[idx[Train_No:], :]

    # trainM = construct_missing_mask(trainX, missing_rate)
    # testM = construct_missing_mask(testX, missing_rate)

    trainM = construct_missing_mask_v2(trainX, missing_rate)
    testM = construct_missing_mask_v2(testX, missing_rate)

    # Train / Test Missing Indicators
    # trainM = Missing[idx[:Train_No], :]
    # testM = Missing[idx[Train_No:], :]

    if save:
        # 保存一些维度数量信息
        fp = dataset_path
        save_dataset_info_fpt = fp + station + '_some_dataset_info.json'
        info_data = [Dim, Train_No, Test_No, missing_rate]
        info_names = ['Dim', 'Train_No', 'Test_No', 'Missing_rate']
        save2json(save_dataset_info_fpt, info_data, info_names)

        # 保存每个站点的归一化参数
        np.save(fp + station + '_MinVal.npy', Min_Val)
        np.save(fp + station + '_MaxVal.npy', Max_Val)

        # 保存Train数据信息
        np.save(fp + station + '_trainX.npy', trainX)
        np.save(fp + station + '_trainM.npy', trainM)
        # 保存Train数据信息
        np.save(fp + station + '_testX.npy', testX)
        np.save(fp + station + '_testM.npy', testM)

    return dict(d=Dim, train_x=trainX, test_x=testX, train_m=trainM, test_m=testM,
                min_val=Min_Val, max_val=Max_Val, train_no=Train_No, test_no=Test_No)


def get_all_users_datasets(args):
    all_stations = args.local_stations
    all_datasets_list = load_datasets(all_stations, args.select_dim)
    all_data_list = []
    for data in all_datasets_list:
        dataset = construct_train_test_dataset(data)
        all_data_list.append(dataset)

    return all_data_list


def generate_saved_datasets(args):
    all_stations = args.local_stations
    all_datasets_list = load_datasets(all_stations, args.select_dim)
    all_data_list = []
    for station, data in zip(all_stations, all_datasets_list):
        print('Station *** {} *** \n'.format(station))
        dataset = construct_train_test_dataset(data, args.missing_rate, True, args.dataset_path, station)
        all_data_list.append(dataset)

    return all_data_list


def generate_different_saved_datasets(args):
    """
    生成不同站点，不同缺失率的数据
    【A, B, C, D, E, F】
    [5%, 10%, 15%, 20%, 25%, 30%]
    :param args:
    :return:
    """
    all_stations = args.local_stations
    all_datasets_list = load_datasets(all_stations, args.select_dim, sample_interval=args.sample_interval)        # quarter_index=args.quarter_index

    all_data_list = []
    missing_rate_list = args.missing_ratios     # 不同缺失率
    load_number_list = args.load_numbers        # 不同载入数量
    for station, data, missing_rate in zip(all_stations, all_datasets_list, missing_rate_list):
        print('Station *** {} *** \n'.format(station))
        dataset = construct_train_test_dataset(data, missing_rate, True, args.dataset_path, station)
        all_data_list.append(dataset)

    save_dataset_info_fpt = args.dataset_path + 'dataset_info.json'
    info_data = [missing_rate_list, load_number_list, args.sample_interval]
    info_names = ['Missing_ratios', 'load_numbers', 'sample_interval']
    save2json(save_dataset_info_fpt, info_data, info_names)

    # 测试数据集可视化
    plot_all_datasets(all_data_list, args.select_dim,
                      args.selected_stations, args.dataset_path)

    return all_data_list


def get_saved_datasets(args):
    selected_stations = args.selected_stations
    all_data_list = []
    fp = args.dataset_path
    for station in selected_stations:
        fpt = fp + station + '_some_dataset_info.json'      # 构造完整路径
        info_json = load_json(fpt)
        Dim = info_json['Dim']
        Train_No = info_json['Train_No']
        Test_No = info_json['Test_No']
        # 归一化参数
        Min_val = np.load(fp + station + '_MinVal.npy')
        Max_val = np.load(fp + station + '_MaxVal.npy')
        # 保存Train数据信息
        trainX = np.load(fp + station + '_trainX.npy')
        trainM = np.load(fp + station + '_trainM.npy')
        # 保存Train数据信息
        testX = np.load(fp + station + '_testX.npy')
        testM = np.load(fp + station + '_testM.npy')

        dataset = dict(d=Dim, train_x=trainX, test_x=testX, train_m=trainM, test_m=testM,
                       min_val=Min_val, max_val=Max_val, train_no=Train_No, test_no=Test_No)
        all_data_list.append(dataset)

    return all_data_list


# 训练和测试数据可视化
def plot_all_datasets(datasets, select_dims, stations, fp):

    labels = stations

    dim_num_train = []
    dim_num_test = []
    for i, station in enumerate(stations):
        s_mdata_train = 1 - datasets[i]['train_m']
        dim_num_train.append(s_mdata_train.sum(axis=0))
        s_mdata_test = 1 - datasets[i]['test_m']
        dim_num_test.append(s_mdata_test.sum(axis=0))

    dim_num_train = np.array(dim_num_train)
    dim_num_test = np.array(dim_num_test)

    # training datasets
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    for j in range(dim_num_train.shape[0]):
        ax.plot(select_dims, dim_num_train[j, :], label=stations[j])

    ax.set_ylabel('Missing Numbers')
    ax.set_title('Missing Numbers for each pollution in training datasets', fontsize=12)
    ax.legend()
    plt.savefig(fp + 'training_datasets.jpg')

    plt.clf()
    plt.close()

    # testing datasets
    fig, ax = plt.subplots()
    for j in range(dim_num_test.shape[0]):
        ax.plot(select_dims, dim_num_test[j, :], label=stations[j])

    ax.set_ylabel('Missing Numbers')
    ax.set_title('Missing Numbers for each pollution in testing datasets', fontsize=12)
    ax.legend()
    plt.savefig(fp + 'testing_datasets.jpg')
    plt.close()


if __name__ == '__main__':
    from param_options import args_parser

    args = args_parser()
    exp_num = 5
    dataset_name = '30_13'
    dataset_path = './constructed_datasets/one_dn({})/'.format(dataset_name)
    # args.dataset_path = './constructed_datasets_6_dq(10_444441_NotRandom)/'
    # 构造数据集
    # all_station_datasets = generate_saved_datasets(args)

    # 构造所有站点不同的缺失率数据
    for exp_n in range(exp_num):
        args.dataset_path = dataset_path + str(exp_n) + '/'
        mkdir(args.dataset_path)
        all_station_datasets = generate_different_saved_datasets(args)

    # 载入数据集（用于测试函数）
    # all_station_datasets = get_saved_datasets(args)

    # 测试数据集可视化
    # plot_all_datasets(all_station_datasets, args.select_dim,
    #                   args.selected_stations, args.dataset_path)

    print(all_station_datasets)



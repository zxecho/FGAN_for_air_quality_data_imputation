import numpy as np
import random
import pandas as pd
from util_tools import save2json, load_json, mkdir
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
            if quarter_index != [1, 2, 3, 4]:  # 如果需要提取个别季节数据的才提取他则不用
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


# 载入数据 v2 ，不需要再将每个站点数据分开载入，每个站点的数据已经合并，并且，这次可以根据参与方和参与方的站点数据不同
# 载入不同构造的数据，用于联邦或单独的本地训练
# PSList 是指每个participants所控制或者具有的站点数据集列表
def load_datasets_v2(PSList, select_dim, return_pd=False, **kwargs):
    # 载入气象站数据
    # 载入数据处理
    dataset_list = []
    pd_dataset_list = []
    q = 0  # 用于标记季节
    si = 0  # 用于采样间隔（h）
    for p_s in PSList:  # p_s： participant stations
        station_data_list = []
        pd_data_list = []
        # 改变日期格式，如果有规定要载入按季节的数据q
        if 'quarter_index' in kwargs.keys():
            quarter_index = kwargs['quarter_index'][q]
            q += 1

        if 'sample_interval' in kwargs.keys():
            interval = kwargs['sample_interval'][si]
            si += 1

        for station in list(p_s):
            data_file = station + '.xlsx'
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
            print('Data %s shape : \n' % station, Data.shape)
            station_data_list.append(Data)
            pd_data_list.append(df_noNaN)
        station_data = np.concatenate(station_data_list, axis=0)
        pd_data = pd.concat(pd_data_list, axis=0)

        if 'quarter_index' in kwargs.keys():
            quarter_data_list = []
            if quarter_index != [1, 2, 3, 4]:  # 如果需要提取个别季节数据的才提取他则不用
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


# 构造缺失mask（包含行，散点）
def construct_missing_mask_v2(data, missing_rate=0.2):
    L = data.shape[0]
    dim = data.shape[1]
    data_shape = data.shape

    row_miss_num = int(2 / 1000 * L)
    column_miss_num = int(3 / 1000 * L)
    # 缺失率
    p_miss = missing_rate

    # 行缺失
    line_missing_p_num = 0  # 行缺失数总量
    while True:
        rand_m1 = np.random.randint(0, L, row_miss_num)
        u = np.unique(rand_m1)
        if len(u) == row_miss_num:
            break

    lm_idx_list = []  # 用于记录随机采取的行缺失index
    for i in range(row_miss_num):
        # 行缺失的维度index
        rand_m1_l = random.sample(range(0, dim + 1), 2)
        lm_idx_list.append(rand_m1_l)
        low = min(rand_m1_l)
        high = max(rand_m1_l)
        line_missing_p_num += (high - low)

    # 列缺失
    rand_m2_v = np.random.randint(0, dim, column_miss_num)
    rand_m2_l = np.random.randint(1, L, column_miss_num)
    # 每个列缺失的数据个数，在一定范围内随机
    missing_len = np.random.randint(2, 6, column_miss_num)
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
    for i in range(column_miss_num):
        if rand_m2_l[i] + missing_len[i] < Missing.shape[0]:
            Missing[rand_m2_l[i]: rand_m2_l[i] + missing_len[i], rand_m2_v[i]] = Missing[rand_m2_l[i]: rand_m2_l[i] +
                                                                                                       missing_len[i],
                                                                                 rand_m2_v[i]] * 0
        else:
            Missing[rand_m2_l[i]:, rand_m2_v[i]] = Missing[rand_m2_l[i]:, rand_m2_v[i]] * 0

    # print('Missing: \n', Missing)
    real_missing_num = 1 - Missing
    real_missing_num = real_missing_num.sum()
    real_missing_rate = real_missing_num / (L * dim)
    print('real missing rate : ', real_missing_rate, 'expect missing rate: ', p_miss)

    return Missing


# 构造缺失mask（只有散点缺失）
def construct_missing_mask_v3(data, missing_rate=0.2):
    L = data.shape[0]
    dim = data.shape[1]
    data_shape = data.shape
    # 缺失率
    p_miss = missing_rate

    # 点缺失
    miss_num = p_miss * L * dim
    prob_missing_p = round(miss_num / (L * dim), 3)
    # 构造missing 矩阵
    Missing = np.zeros(data_shape)
    p_miss_vec = prob_missing_p * np.ones((L, 1))
    for i in range(dim):
        A = np.random.uniform(0., 1., size=[L, ])
        B = A > p_miss_vec[i]
        Missing[:, i] = 1. * B

    # print('Missing: \n', Missing)
    real_missing_num = 1 - Missing
    real_missing_num = real_missing_num.sum()
    real_missing_rate = real_missing_num / (L * dim)
    print('real missing rate : ', real_missing_rate, 'expect missing rate: ', p_miss)

    return Missing


# 构造缺失数据
def construct_train_test_dataset(Data, missing_rate=0.2, save=False,
                                 dataset_path='', station='', p_info=(0, 0), random=True):
    # 函数参数
    """
    :param missing_rate:
    :param dataset_path:
    :param Data: 构造的数据L
    :param save: 是否保存
    :param station: 站点名称
    :return:
    """
    # 选择制定数量数据
    No = Data.shape[0]

    # 数据参数
    Dim = Data.shape[1]  # 数据维度

    # 归一化
    # Normalization (0 to 1)
    Min_Val = np.zeros(Dim)
    Max_Val = np.zeros(Dim)

    # mini-max normalization
    for i in range(Dim):
        Min_Val[i] = np.min(Data[:, i])
        Max_Val[i] = np.max(Data[:, i])
        if i == 3:
            Data[:, i] = Data[:, i] - np.min(Data[:, i])
            Data[:, i] = Data[:, i] / (Max_Val[i] - Min_Val[i] + 1e-8)

    # mean-std normalization
    # for i in range(Dim):
    #     Min_Val[i] = np.std(Data[:, i])
    #     Max_Val[i] = np.mean(Data[:, i])
    #     Data[:, i] = Data[:, i] - Max_Val[i]
    #     Data[:, i] = Data[:, i] / (Min_Val[i] + 1e-8)

    # Missing = construct_missing_mask_v2(Data, missing_rate)

    # 分离出训练和验证的数据集
    if random:
        idx = np.random.permutation(No)
        Train_No = int(No * train_rate)
        Test_No = No - Train_No
        # Train / Test Features
        trainX = Data[idx[:Train_No], :]
        testX = Data[idx[Train_No:], :]
        # trainM = Missing[idx[:Train_No], :]
        # testM = Missing[idx[Train_No:], :]
    else:
        tr_i = int(train_rate / (1 - train_rate))
        Test_number = set([i for i in range(No) if i % (tr_i + 1) == 0])
        Train_number = set([i for i in range(No)]) - Test_number
        # Train / Test Features
        trainX = Data[list(Train_number), :]
        testX = Data[list(Test_number), :]
        Train_No = len(Train_number)
        Test_No = len(Test_number)
        # trainM = Missing[list(Train_number), :]
        # testM = Missing[list(Test_number), :]
    # V1版本
    # trainM = construct_missing_mask(trainX, missing_rate)
    # testM = construct_missing_mask(testX, missing_rate)
    # V2版本
    trainM = construct_missing_mask_v2(trainX, missing_rate)
    testM = construct_missing_mask_v2(testX, missing_rate)
    # 最初的版本
    # Train / Test Missing Indicators
    # Missing = construct_missing_mask_v2(Data)
    # trainM = Missing[idx[:Train_No], :]
    # testM = Missing[idx[Train_No:], :]

    if save:
        # 保存一些维度数量信息
        fp = dataset_path
        data_num = station + '{}{}'.format(p_info[0], p_info[1])
        save_dataset_info_fpt = fp + data_num + '_some_dataset_info.json'
        info_data = [Dim, Train_No, Test_No, missing_rate]
        info_names = ['Dim', 'Train_No', 'Test_No', 'Missing_rate']
        save2json(save_dataset_info_fpt, info_data, info_names)

        # 保存每个站点的归一化参数
        np.save(fp + data_num + '_MinVal.npy', Min_Val)
        np.save(fp + data_num + '_MaxVal.npy', Max_Val)

        # 保存Train数据信息
        np.save(fp + data_num + '_trainX.npy', trainX)
        np.save(fp + data_num + '_trainM.npy', trainM)
        # 保存Train数据信息
        np.save(fp + data_num + '_testX.npy', testX)
        np.save(fp + data_num + '_testM.npy', testM)

    return dict(d=Dim, train_x=trainX, test_x=testX, train_m=trainM, test_m=testM,
                min_val=Min_Val, max_val=Max_Val, train_no=Train_No, test_no=Test_No)


# 构造缺失数据，用所有数据作为训练集
def construct_dataset_vall(Data, missing_rate=0.2, save=False,
                           dataset_path='', station='', p_info=(0, 0), random=True):
    # 函数参数
    """
    :param p_info:
    :param missing_rate: 缺失率
    :param dataset_path: 数据路径
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

    Missing = construct_missing_mask_v2(Data, missing_rate)

    if save:
        # 保存一些维度数量信息
        fp = dataset_path
        data_num = station + '{}{}'.format(p_info[0], p_info[1])
        save_dataset_info_fpt = fp + data_num + '_some_dataset_info.json'
        info_data = [Dim, No, missing_rate]
        info_names = ['Dim', 'No', 'Missing_rate']
        save2json(save_dataset_info_fpt, info_data, info_names)

        # 保存每个站点的归一化参数
        np.save(fp + data_num + '_MinVal.npy', Min_Val)
        np.save(fp + data_num + '_MaxVal.npy', Max_Val)

        # 保存Train数据信息
        np.save(fp + data_num + '_trainX.npy', Data)
        np.save(fp + data_num + '_trainM.npy', Missing)

    return dict(d=Dim, train_x=Data, test_x=Data, train_m=Missing, test_m=Missing,
                min_val=Min_Val, max_val=Max_Val, train_no=No, test_no=No)


def get_saved_datasets_vall(args):
    selected_stations = args.selected_stations
    num_d = args.num_d
    all_data_list = []
    fp = args.dataset_path

    for i, s in enumerate(selected_stations):
        data_num = s + '{}{}'.format(i, num_d[i])
        fpt = fp + data_num + '_some_dataset_info.json'  # 构造完整路径
        info_json = load_json(fpt)
        Dim = info_json['Dim']
        No = info_json['No']

        # 归一化参数
        Min_val = np.load(fp + data_num + '_MinVal.npy')
        Max_val = np.load(fp + data_num + '_MaxVal.npy')
        # 保存Train数据信息
        X = np.load(fp + data_num + '_trainX.npy')
        M = np.load(fp + data_num + '_trainM.npy')

        dataset = dict(d=Dim, train_x=X, test_x=X, train_m=M, test_m=M,
                       min_val=Min_val, max_val=Max_val, train_no=No, test_no=No)
        all_data_list.append(dataset)

    return all_data_list


# 训练和测试数据可视化
def plot_all_datasets(datasets, select_dims, stations, fp):

    # 针对load_dataset_v2，对不同参与方，每个参与方都有自己的站点数据添加
    if type(args.selected_stations[0]) == list:
        stations = args.clients
    else:
        stations = args.selected_stations

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


def generate_selected_station_save_datasets(args, if_random=False):
    """
    生成选定的站点，不同缺失率的数据集
    :param args:
    :return:
    """
    selected_stations = args.selected_stations
    num_p = len(selected_stations)
    num_d = args.num_d
    all_datasets_list = load_datasets_v2(selected_stations, args.select_dim,
                                         sample_interval=args.sample_interval)  # quarter_index=args.quarter_index

    all_data_list = []
    missing_rate_list = args.missing_ratios  # 不同缺失率
    load_number_list = args.load_numbers  # 不同载入数量

    for p, nd, station, data, missing_rate in zip(range(num_p), num_d,
                                                  selected_stations, all_datasets_list, missing_rate_list):
        print('Station *** {} {}*** \n'.format(station, p))
        # 针对load_dataset_v2，对不同参与方，每个参与方都有自己的站点数据添加
        if type(station) == list:
            s_name = ''
            for si in range(len(station)):
                s_name += station[si]
            station = s_name
        dataset = construct_train_test_dataset(data, missing_rate, True, args.dataset_path, station,
                                               p_info=(p, nd), random=if_random)
        # dataset = construct_dataset_vall(data, missing_rate, True, args.dataset_path, station,
        #                                  p_info=(p, nd), random=if_random)
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
    num_d = args.num_d
    all_data_list = []
    fp = args.dataset_path

    for i, s in enumerate(selected_stations):
        # 针对load_dataset_v2，对不同参与方，每个参与方都有自己的站点数据添加
        if type(s) == list:
            s_name = ''
            for si in range(len(s)):
                s_name += s[si]
            s = s_name
        data_num = s + '{}{}'.format(i, num_d[i])
        fpt = fp + data_num + '_some_dataset_info.json'  # 构造完整路径
        info_json = load_json(fpt)
        Dim = info_json['Dim']
        Train_No = info_json['Train_No']
        Test_No = info_json['Test_No']
        # 归一化参数
        Min_val = np.load(fp + data_num + '_MinVal.npy')
        Max_val = np.load(fp + data_num + '_MaxVal.npy')
        # 保存Train数据信息
        trainX = np.load(fp + data_num + '_trainX.npy')
        trainM = np.load(fp + data_num + '_trainM.npy')
        # 保存Train数据信息
        testX = np.load(fp + data_num + '_testX.npy')
        testM = np.load(fp + data_num + '_testM.npy')

        dataset = dict(d=Dim, train_x=trainX, test_x=testX, train_m=trainM, test_m=testM,
                       min_val=Min_val, max_val=Max_val, train_no=Train_No, test_no=Test_No)
        all_data_list.append(dataset)

    return all_data_list


if __name__ == '__main__':
    from param_options import args_parser

    args = args_parser()
    exp_num = 5

    generate_condition = 'one_time'  # general /  one_time / missing_rate
    dataset_name_temp = '(A5_A10_A15)_nCO_321r'

    params_test_list = []
    test_param_name = None

    if generate_condition == 'missing_rate':
        params_test_list = [5, 10, 15, 20, 25, 30, 35, 40]
        test_param_name = 'missing_rate'
        Dname_prefix = '(A{})'
        Dname_suffix = 'norm_1r_1P'
    elif generate_condition == 'one_time':
        params_test_list = [1]
        test_param_name = 'One_time'

    for param in params_test_list:
        if generate_condition == 'missing_rate':
            args.missing_ratios = [param / 100] * len(args.selected_stations)
            dataset_name = Dname_prefix.format(param, param, param) + '_' + Dname_suffix
            print('===================== Missing ratio {}========================'.format(param))
        elif generate_condition == 'one_time':
            dataset_name = dataset_name_temp + '_' + test_param_name

        dataset_path = './constructed_datasets/{}/'.format(dataset_name)
        # 构造所有站点不同的缺失率数据
        for exp_n in range(exp_num):
            args.dataset_path = dataset_path + str(exp_n) + '/'
            mkdir(args.dataset_path)
            all_station_datasets = generate_selected_station_save_datasets(args, if_random=True)

            # 载入数据集（用于测试函数）
            # all_station_datasets = get_saved_datasets(args)

            print(all_station_datasets)

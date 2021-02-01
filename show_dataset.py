import numpy as np
import random
import pandas as pd
from util_tools import save2json, load_json, mkdir, normalization
import matplotlib.pyplot as plt
import os
from param_options import args_parser

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 对数据进行行和列缺失统计
def missing_data_statistic(data_shape, data_mask):
    # 建立行缺失的数据记录结构
    row_missing = {k: 0 for k in range(2, data_shape[1] + 1)}
    colum_missing = {k: 0 for k in range(2, 35)}
    # 检查行的连续缺失
    for i in range(data_shape[0]):
        j = 0
        print('第 {} 行数据分析'.format(i))
        if data_mask[i, :].sum() == data_shape[1]:
            continue
        while j < data_shape[1]:
            missing_len = 0
            while data_mask[i, j] == 0:
                missing_len += 1
                j += 1
                if j >= data_shape[1]:
                    break
            j += 1
            if missing_len > 1:
                row_missing[missing_len] += 1
    # 统计列缺失情况
    for i in range(data_shape[1]):
        j = 0
        print('第 {} 列数据分析'.format(i))
        if data_mask[:, i].sum() == data_shape[1]:
            continue
        while j < data_shape[0]:
            missing_len = 0
            while data_mask[j, i] == 0:
                missing_len += 1
                j += 1
                if j >= data_shape[0]:
                    break
            j += 1
            if missing_len > 1:
                colum_missing[missing_len] += 1

    return row_missing, colum_missing


# 对缺失数据统计结果进行可视化
def plot_missing_data_statistics(dataset_name_l, item_missing_stat_l, data_missing_number_l,
                                 r_stat_l, c_stat_l, data_len_l, select_dim):
    # 用于汇总行和列每个站点数据集有多少出连续缺失
    r_stat_sum_l = []
    c_stat_sum_l = []
    # 用于汇总行和列每个站点数据集有多少数据点是连续缺失
    c_mp_num_l = []

    # 表示一个站点数据集总的缺失率
    fig, ax = plt.subplots()
    y = [data_missing_number_l[i] / (data_len_l[i] * len(select_dim)) * 100 for i in range(len(dataset_name_l))]
    ax.plot(dataset_name_l, y)
    ax.tick_params(labelsize=13)
    ax.set_xlabel('Statioin', size=13)
    ax.set_ylabel('Missing rate(%)', size=13)
    # ax.set_title('Missing rate of each station')

    save_path = 'E:/zx/Fed-AQ/experiments_results/Figures/Dataset_overview_v2/'
    #  save_path = save_path + dataset_name + '/'
    mkdir(save_path)
    plt.savefig(save_path + 'missing_rate' + '.eps')
    plt.savefig(save_path + 'missing_rate' '.svg')
    # 保存到本地json
    save2json(save_path + 'missing_rate.json', y, dataset_name_l)

    # 每个污染物成分的缺失分析
    plt.style.use('ggplot')
    fig1, ax1 = plt.subplots()
    for dataset_name, item_missing_stat, data_len in zip(dataset_name_l, item_missing_stat_l, data_len_l):
        y = [item_missing_stat[d] / data_len * 100 for d in select_dim]
        ax1.plot(select_dim, y, label=dataset_name)

        # 保存到本地json
        save2json(save_path + dataset_name + '_pollutants.json', y, select_dim)

        ax1.tick_params(labelsize=13)
        ax1.set_xlabel('Pollutants', size=13)
        ax1.set_ylabel('Missing rate(%)', size=13)
        # ax1.set_title('Missing rate of each pollutant')
        ax1.legend()

    # save_path = 'E:/zx/Fed-AQ/experiments_results/Figures/Dataset_overview/'
    #  save_path = save_path + dataset_name + '/'
    # mkdir(save_path)
    plt.savefig(save_path + 'pollutants_stat' + '.eps')
    plt.savefig(save_path + 'pollutants_stat' '.svg')

    # 行缺失
    fig2, ax2 = plt.subplots()
    for i, dataset_name, r_stat in zip(range(len(dataset_name_l)), dataset_name_l, r_stat_l):
        # bar graphs
        y = [v for v in r_stat.values()]
        x = [k for k in r_stat.keys()]

        y = np.array(y)
        x = np.array(x)

        width = 0.25
        ax2.bar(x + width * i, y, width, label=dataset_name)
        ax2.tick_params(labelsize=13)
        ax2.set_xlabel('Gap lengths', size=13)
        ax2.set_ylabel('Number', size=13)
        # ax2.set_title(' Row-wise area')
        ax2.set_xticks(x + width * i / 2)
        ax2.set_xticklabels(x)
        ax2.legend()
        r_stat_sum = np.sum(y)  # 总共多少个行缺失
        r_stat_sum_l.append(r_stat_sum)
    save_name = save_path + '/Row-wise-missing-data-stat'
    plt.savefig(save_name + '.eps')
    plt.savefig(save_name + '.svg')

    # 列缺失
    fig3, ax3 = plt.subplots()
    for i, dataset_name, c_stat in zip(range(len(dataset_name_l)), dataset_name_l, c_stat_l):
        # bar graphs
        # y = [v for v in c_stat.values()]
        # x = [k for k in c_stat.keys()]

        x_split = ['2', '3', '4', '5', '6-15', '16-30', '>30']
        y_split = [0 for _ in range(len(x_split))]

        # 对列缺失进行段落划分，否则统计显示很不方便
        c_missing_num = 0
        for x_, y_ in zip(c_stat.keys(), c_stat.values()):
            c_missing_num += x_ * y_
            if x_ == 2:
                y_split[0] += y_
            if x_ == 3:
                y_split[1] += y_
            if x_ == 4:
                y_split[2] += y_
            if x_ == 5:
                y_split[3] += y_
            if 5 < x_ <= 15:
                y_split[4] += y_
            if 15 < x_ <= 30:
                y_split[5] += y_
            if 30 < x_:
                y_split[6] += y_

        y = np.array(y_split)
        x = np.array(np.arange(len(x_split)))

        width = 0.25
        ax3.bar(x + width * i, y, width, label=dataset_name)
        ax3.tick_params(labelsize=13)
        ax3.set_xlabel('Gap lengths', size=13)
        ax3.set_ylabel('Number', size=13)
        # ax3.set_title(' Column-wise area')
        ax3.set_xticks(x + width * i / 2)
        ax3.set_xticklabels(x_split)
        plt.xticks(rotation=-15)  # 设置x轴标签旋转角度
        plt.tick_params(axis='x', labelsize=10)  # 设置x轴标签大小
        ax3.legend()
        c_stat_sum = np.sum(y)  # 总共多少个列缺失
        c_stat_sum_l.append(c_stat_sum)
        c_mp_num_l.append(c_missing_num)
    save_name = save_path + '/Column-wise-missing-data-stat'
    plt.savefig(save_name + '.eps')
    plt.savefig(save_name + '.svg')

    # 行和列缺失的总的个数
    fig4, ax4 = plt.subplots()
    for i, dataset_name, r_stat_sum, c_stat_sum in zip(range(len(dataset_name_l)), dataset_name_l, r_stat_sum_l,
                                                       c_stat_sum_l):
        y = [r_stat_sum, c_stat_sum]
        x_ticklable = ['row', 'column']
        x = np.arange(len(x_ticklable))
        width = 0.25
        ax4.bar(x + width * i, y, width, label=dataset_name)
        ax4.tick_params(labelsize=13)
        ax4.set_ylabel('Number', size=13)
        # ax4.set_title('Numbers of missing row and column')
        ax4.set_xticks(x + width * i / 2)
        ax4.set_xticklabels(x_ticklable)
        ax4.legend()
    save_name = save_path + '/Sum-missing-data-stat'
    plt.savefig(save_name + '.eps')
    plt.savefig(save_name + '.svg')

    # 行和列缺失的总的占数据集的比例
    fig5, ax5 = plt.subplots()
    for i, dataset_name, r_stat_sum, c_missing_num, data_len in zip(range(len(dataset_name_l)), dataset_name_l,
                                                                    r_stat_sum_l, c_mp_num_l, data_len_l):
        y = [r_stat_sum / data_len * 100, c_missing_num / (data_len * len(select_dim)) * 100]
        x_ticklable = ['row', 'column']
        x = np.arange(len(x_ticklable))
        width = 0.25
        ax5.plot(x_ticklable, y, label=dataset_name)
        # 保存到本地json
        save2json(save_path + dataset_name + '_rc.json', y, x_ticklable)
        ax5.tick_params(labelsize=13)
        ax5.set_ylabel('Ratio(%)', size=13)
        # ax5.set_title('Ratio of missing data in row and column gaps')
        # ax5.set_xticks(x + width * i / 2)
        # ax5.set_xticklabels(x_ticklable)
        ax5.legend()
    save_name = save_path + '/Ratio-of-missing-data-stat_for_r&c'
    plt.savefig(save_name + '.eps')
    plt.savefig(save_name + '.svg')

    # plt.show()


# 载入数据
def datasets_satistical(dataset_name, AQS, select_dim, **kwargs):
    # 载入气象站数据
    # 载入数据处理
    dataset_list = []
    # 用于保存处理之后的站点数据
    dataset_name_list = []
    source_data_df_NaN_list = []
    source_data_missing_number = []  # 用于记录缺失量
    station_r_list = []
    station_c_list = []
    dataset_length_list = []

    for station in AQS:
        station_data_list = []
        source_data_list = []

        for i in range(1, 3):
            data_file = station + '_' + str(i) + '.xlsx'
            print(data_file)
            data_file_path = './{}/'.format(dataset_name) + data_file
            air_data_df = pd.read_excel(data_file_path)

            df_sel = air_data_df.loc[:, select_dim]
            df_noNaN = df_sel[df_sel.notnull().sum(axis=1) == len(select_dim)]
            Data = df_noNaN.values
            print('Data %d shape : \n' % i, Data.shape)
            station_data_list.append(Data)
            source_data_list.append(df_sel)
        station_data = np.concatenate((station_data_list[0], station_data_list[1]), axis=0)
        source_data_df = pd.concat(source_data_list, axis=0)  # 一个站点的一年完整数据

        # 对一个完整的station数据进行统计
        source_data_df_NaN = source_data_df.isnull().sum()
        source_data_df_notNaN_mask = source_data_df.notnull().values
        mask_shape = source_data_df_notNaN_mask.shape
        station_r, station_c = missing_data_statistic(mask_shape, source_data_df_notNaN_mask)

        dataset_name_list.append(station)
        source_data_df_NaN_list.append(source_data_df_NaN)
        source_data_missing_number.append(source_data_df.isnull().values.sum())
        station_r_list.append(station_r)
        station_c_list.append(station_c)
        dataset_length_list.append(mask_shape[0])

        dataset_list.append(station_data)
        print('station_data shape: ', station_data.shape)

    plot_missing_data_statistics(dataset_name_list, source_data_df_NaN_list, source_data_missing_number,
                                 station_r_list, station_c_list, dataset_length_list, select_dim)
    return dataset_list


if __name__ == '__main__':
    args = args_parser()
    dataset_name = 'air_quality_datasets'
    datasets_satistical(dataset_name, args.selected_stations, args.select_dim, )

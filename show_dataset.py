import numpy as np
import random
import pandas as pd
from util_tools import save2json, load_json, mkdir, normalization
import matplotlib.pyplot as plt
import os
from param_options import args_parser
from Fed_GAIN.plot_indexes_resluts import get_all_datasets
from util_tools import mkdir, compute_avg_of_data_in_file

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


def plot_indicator_avg_results_m(logdir, save_path, xaxis, value, save_csv_fpth, leg=None,
                                 condition=None, select_dim=None):
    # 重新实现了seaborn的带有均值线的多次实验结果图
    if leg is None:
        leg = ['Federated GAN', 'Local GAN']

    datas = get_all_datasets(logdir, leg)

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
    if value == 'all_rmse':
        # 不同的condition
        # 创建绘图
        fig, ax = plt.subplots()
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

            save_avg_csv_pt = save_csv_fpth + mode + '_' + value + '_avg_resluts.csv'
            mode_save_data = {'RMSE': indicator_avg, 'std': indicator_std}
            # 使用pandas保存成csv
            dataframe = pd.DataFrame(mode_save_data)
            dataframe.to_csv(save_avg_csv_pt, index=True, sep=',')
            # save_all_avg_results(fed_save_csv_pt, indicator_avg, [value], xaxis)
    else:
        # 创建绘图
        fig, ax = plt.subplots(2, 3, figsize=(19.2, 10.8))
        plt.subplots_adjust(top=0.95)
        for p, pollution in enumerate(select_dim):
            # 不同的condition
            for mode in condition_sets:
                avg_t = 0
                # 计算被标记的不同unit之间的均值
                condition_min_list = []
                condition_max_list = []
                condition_unit_data_list = []
                condition_data = data[data.Condition == mode]
                for u in unit_sets:
                    condition_unit_data = condition_data[condition_data.Unit == u][pollution].values
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
                    ax.set_ylabel(value.upper(), size=13)
                    ax.set_xlabel('Participant', size=13)
                    ax.set_title(pollution)
                    ax.legend()
                    ax.margins(x=0)

                    return est, sd

                # 将几次的实验数据进行拼接，形成一个（station， exp_time）形状的数组
                all_condition_unit_data = np.concatenate(condition_unit_data_list, axis=0)
                xaxis_from_sets = [i for i in xaxis_sets]

                indicator_avg, indicator_std = tsplot(ax[p // 3, p % 3], xaxis_from_sets, all_condition_unit_data)

                save_avg_csv_pt = save_csv_fpth + mode + '_' + pollution + '_avg_resluts.csv'
                mode_save_data = {'RMSE': indicator_avg, 'std': indicator_std}
                # 使用pandas保存成csv
                dataframe = pd.DataFrame(mode_save_data)
                dataframe.to_csv(save_avg_csv_pt, index=True, sep=',')

    save_path = save_path + value + '_results'
    plt.savefig(save_path + '.svg')

    plt.close()


# 根据不同污染物的结果画出图像
def plot_eval_results_by_different_pollution():
    pass


def plot_fed_avg_acc(model_files_path, indicator, fig_save_path):
    for files_path in model_files_path:
        filep_l = os.listdir(files_path)
        print(filep_l)
        # 创建绘图
        fig, ax = plt.subplots()
        datafile_list = []

        for pdir in filep_l:
            file_path = os.path.join(files_path, pdir)
            datafile = pd.read_csv(file_path)
            datafile_list.append(np.array(datafile.values[0][1:], dtype=np.float).reshape(1, -1))
            x_axis_index = datafile.keys().tolist()
            x_axis_index = x_axis_index[1:]

        # 用于替代seaborn中的tsplot绘图函数
        def tsplot(ax, x, data, label_name='', **kw):
            est = np.mean(data, axis=0)
            sd = np.std(data, axis=0)
            cis = (est - sd, est + sd)

            ax.yaxis.grid(True)
            # ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
            # ax.plot(x, est, label=label_name, **kw)
            plt.boxplot(data)
            ax.set_xticks([y + 1 for y in range(len(x))], )
            # add x-tick labels
            plt.setp(ax, xticks=[y + 1 for y in range(len(x))],
                     xticklabels=x)
            ax.tick_params(labelsize=13)
            ax.set_ylabel('Test RMSE', size=13)
            ax.set_xlabel('Pollution', size=13)
            ax.margins(x=0)

            return est, sd

        xaxis_from_sets = []
        all_data = np.concatenate(datafile_list, axis=0)
        indicator_avg, indicator_std = tsplot(ax, x_axis_index, all_data, label_name='')
        # 保存到本地
        fp_splits = files_path.split('/')
        save_avg_csv_pt = os.path.join(*fp_splits[:-1]) + indicator + '_avg_resluts.csv'
        mode_save_data = {'avg': indicator_avg, 'std': indicator_std}

        # 使用pandas保存成csv
        dataframe = pd.DataFrame(mode_save_data, index=x_axis_index)
        dataframe.to_csv(save_avg_csv_pt, index=True, sep=',')
        # ax.plot(x_axis_index, [96.4]*100, label='None-fed', linestyle='dashed')
        plt.savefig(fig_save_path + '{}_avg_results_boxplot.svg'.format(indicator))


def show_fed_eval_results(args, exp_name, results_saved_file, results_plot_file, cross_validation_sets):
    result_save_root = './{}/'.format(results_saved_file) + exp_name + '/'
    plots_save_root = './{}/'.format(results_plot_file) + exp_name + '/'
    indicator_list = ['rmse', 'd2', 'r2', 'all_rmse']
    leg = ['Federated', 'Independent']

    for indicator_name in indicator_list:
        # 建立保存结果的文件夹
        indicator_avg_results_csv_save_fpth = result_save_root + 'avg_' + indicator_name + '/'
        for mode in leg:
            mkdir(indicator_avg_results_csv_save_fpth + mode + '/')

        # 计算每个数据集的几次实验的均值
        for c in range(cross_validation_sets):
            results_logdir = [result_save_root + 'datasets_' + str(c) + '/' + indicator_name + '/' + model_name + '/'
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
                                     select_dim=arg.select_dim)
        # plot_fed_avg_acc(results_logdir, indicator_name, fig_save_path)
        # plot_indicator_results(results_logdir, fig_save_path, indicator_name)

        print("\033[0;34;40m >>>[Visulize] Finished save {} resluts figures! \033[0m".format(indicator_name))


if __name__ == '__main__':
    arg = args_parser()

    # 做实验
    exp_total_time = 1
    cross_validation_set_num = 5

    result_saved_file = 'Fed_wGAN_results'
    result_plot_file = 'Fed_wGAN_plot_results_test'

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

        show_fed_eval_results(arg, exp_name, result_saved_file, result_plot_file, cross_validation_set_num)

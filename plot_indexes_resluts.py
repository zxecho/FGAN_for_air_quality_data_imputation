import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import os.path as osp
import seaborn as sns
from param_options import args_parser

# Plot results 绘制训练结果
DIV_LINE_WIDTH = 50
units = dict()


def plot_test_csv(fed_csv, idpt_csv, title=''):
    fed_f = pd.read_csv(fed_csv, sep=',')
    idpt_f = pd.read_csv(idpt_csv, sep=',')
    clumns = fed_f.shape[1] - 1
    rows = fed_f.shape[0]
    fed_f_array = np.array(fed_f)
    idpt_f_array = np.array(idpt_f)
    fig, ax = plt.subplots(nrows=1, ncols=clumns)
    for i in range(clumns):
        ax[i].plot(fed_f_array[:, 0], fed_f_array[:, i+1], label='Fed')
        ax[i].plot(idpt_f_array[:, 0], idpt_f_array[:, i+1], label='Independent')
        ax[i].legend(loc='lower right')
        ax[i].set_title(fed_f.columns[i+1])

    plt.show()


def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition=None, **kwargs):

    if isinstance(data, list):
        data = pd.concat(data)
    fig = plt.figure()
    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)


def get_datasets(logdir, file_suffix='', condition=None):
    """
    file_suffix: 文件后缀，例如“results.txt, results.csv”
    """
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        print('files: ', files)
        for file in files:
            if condition not in units:
                units[condition] = 0
            unit = units[condition]
            units[condition] += 1
            try:
                if file_suffix[-3:] == 'txt':
                    exp_data = pd.read_table(os.path.join(root, file))
                if file_suffix[-3:] == 'csv':
                    exp_data = pd.read_csv(os.path.join(root, file))
            except:
                print('Could not read from %s' % os.path.join(root, file))
                continue
            xaxis = [i+1 for i in range(exp_data.shape[0])]
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition', condition)
            if 'station' not in exp_data.columns:
                exp_data.insert(len(exp_data.columns), 'station', xaxis)
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and (logdir[-1] == os.sep or logdir[-1] == '/'):
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    # 这里是确保legend的数量和要绘制的logdir的数量一样
    assert not legend or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # 从具体的logdirs载入数据
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, 'results.csv', leg)
    else:
        for log in logdirs:
            data += get_datasets(log, 'results.csv')
    return data


def make_plots_(all_logdirs, legend=None, xaxis=None, values=None, count=False,
                font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean'):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator)
    plt.show()


def make_average_plots(logdirs, plot_values, legs):
    """
    不同的情况运行多次，取多次均值，画出均值线，其他为虚色背景

    NOTE: 需要注意的是，需要手工的把运行的几组放到同一个文件夹下，然后转入该文件夹相对路径
    """
    data = []
    for log in logdirs:
        data += get_datasets(log, log[7:])

    # data = get_all_datasets(logdirs)
    xaxis = 'Epoch'
    value = plot_values
    condition = 'Condition1'
    smooth = 1
    estimator = getattr(np, 'mean')

    plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator)
    plt.show()


def plot_indexes_result():
    # 用于绘制一次的结果
    fed_mse_csv = './results/6_20_100_32/4/fed_mse_test_results.csv'
    idpt_mse_csv = './results/6_20_100_32/4/idpt_mse_test_results.csv'
    plot_test_csv(fed_mse_csv, idpt_mse_csv, 'RMSE')
    fed_d2_csv = './results/6_20_100_32/4/fed_d2_test_results.csv'
    idpt_d2_csv = './results/6_20_100_32/4/idpt_d2_test_results.csv'
    plot_test_csv(fed_d2_csv, idpt_d2_csv, 'D2')
    fed_r2_csv = './results/6_20_100_32/4/fed_r2_test_results.csv'
    idpt_r2_csv = './results/6_20_100_32/4/idpt_r2_test_results.csv'
    plot_test_csv(fed_r2_csv, idpt_r2_csv, 'R2')
    plt.show()


def plot_indicator_results(logdir, save_path, value, xaxis='station', leg=None):
    if leg is None:
        leg = ['Fed', 'Independent']

    datas = get_all_datasets(logdir, leg)
    # print(datas)
    condition = 'Condition'
    smooth = 1
    estimator = getattr(np, 'mean')

    plot_data(datas, xaxis=xaxis, value=value, condition=condition, estimator=estimator)

    save_path = save_path + value + '_results'
    plt.savefig(save_path, format='eps')
    plt.savefig(save_path, format='svg')

    plt.show()


if __name__ == '__main__':
    args = args_parser()
    exp_name = 'FedWeightAvg(load_numbers_soft)_(5-1)_dn_100_32(lrS_Adam_Fed_0.001_100_0.001_100_bs_128)_32'
    result_save_file = './results/' + exp_name + '/'
    # plot_indexes_result()

    # plot_indexes_results()
    # 绘制测试结果图像
    print('====================== Save every component indicator result ========================')
    for indicator in ['rmse', 'd2', 'r2']:
        print('{} results'.format(indicator))
        for component in args.select_dim:
            results_logdir = [result_save_file + indicator + '/' + model_name + '/' for model_name in ['fed', 'idpt']]
            fig_save_path = result_save_file + indicator + '/'
            plot_indicator_results(results_logdir, fig_save_path, component)

    print(">>> Finished save resluts figures!")

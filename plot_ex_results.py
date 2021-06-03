import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from param_options import args_parser
from Fed_gain_test_run import plot_indicator_avg_results_m
from util_tools import mkdir
plt.style.use('seaborn')


def plot_local_indicator_avg_results(logdirs, save_path, save_name='', leg=None, fig_title='', file_type='excel'):
    # 重新实现了seaborn的带有均值线的多次实验结果图
    if leg is None:
        leg = ['RMSE']

    # 创建绘图
    fig, ax = plt.subplots()

    for logdir, l in zip(logdirs, leg):
        if file_type == 'excel':
            # 读取excel 文件
            df = pd.read_excel(logdir, sheet_name=0)
        elif file_type == 'csv':
            df = pd.read_csv(logdir)

        xaxis = np.array(df['Index'])
        est = np.array(df['RMSE'])
        std = np.array(df['std'])

        # 用于替代seaborn中的tsplot绘图函数
        def tsplot(ax, x, est, std, **kw):
            cis = (est - std, est + std)
            ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
            ax.plot(x, est, label=l, **kw)
            ax.tick_params(labelsize=15)
            ax.set_ylabel('RMSE', size=15)
            # ax.set_xlabel('Missing rate(%)', size=13)
            ax.set_xlabel('Missing rate', size=15)
            ax.set_title(fig_title)
            ax.legend()
            ax.margins(x=0)

        # 绘图
        tsplot(ax, xaxis, est, std)

    # save_all_avg_results(fed_save_csv_pt, indicator_avg, [value], xaxis)
    save_path = save_path + save_name
    plt.savefig(save_path + '.eps')
    plt.savefig(save_path + '.svg')

    plt.close()
    print('Finished saving figure!')


def plot_federated_indicator_avg_results(dataset_number):
    args = args_parser()
    # dataset_number = 'one_mi_v1((A5)_1)'

    results_saved_file = 'results_one_dn'
    results_plot_file = 'plot_results_one_dn'
    exp_name = '{}_latest'.format(dataset_number, )

    result_save_root = './{}/'.format(results_saved_file) + exp_name + '/'
    plots_save_root = './{}/'.format(results_plot_file) + exp_name + '/'
    indicator_name = 'all_rmse'
    leg = ['Fed', 'Independent']

    results_logdir = [result_save_root + 'avg_' + indicator_name + '/' + model_name + '/' for model_name in leg]
    fig_save_path = plots_save_root + indicator_name + '/'
    csv_save_fpth = result_save_root + 'avg_' + indicator_name + '/'
    mkdir(fig_save_path)

    # 将每个数据集计算的均值再计算总的均值和绘制方差均值线图
    plot_indicator_avg_results_m(results_logdir, fig_save_path, 'station', indicator_name, csv_save_fpth)


if __name__ == '__main__':
    exp_model = 'Local'   # Local / Fed / Fed_vs_Local

    if exp_model == 'Local':
        data_dir = ['E:/zx/Fed-AQ/experiments_results/wGAI_norCO_one_mi(A_1r).xlsx',
                    'E:/zx/Fed-AQ/experiments_results/GAIN_norCO_one_mi(A_1r).xlsx',
                    'E:/zx/Fed-AQ/experiments_results/EM_one_mi_v1((A)_1r).xlsx',
                    'E:/zx/Fed-AQ/experiments_results/Spline_one_mi_v1((A)_1r).xlsx',
                    # 'E:/zx/Fed-AQ/experiments_results/Cubic_one_mi_v1((A)_1r).xlsx',
                    # 'E:/zx/Fed-AQ/experiments_results/KNN_one_mi_v1((A)_1r).xlsx',
                    'E:/zx/Fed-AQ/experiments_results/Linear_one_mi_v1((A)_1r).xlsx',
                    'E:/zx/Fed-AQ/experiments_results/MultiBayesian_one_mi_v1((A)_1r).xlsx'
                    ]
        leg = ['Ours', 'GAIN', 'EM',  'Spline', 'Linear', 'MICE']      # 'MICE'
        save_path = 'E:/zx/Fed-AQ/experiments_results/Figures/'
        figure_name = 'one_mi(A_1r)_on_various_algos_latest'
        plot_local_indicator_avg_results(data_dir, save_path=save_path, save_name=figure_name, leg=leg)
    elif exp_model == 'Fed':
        # dataset_number = 'one_mi((A5_B10_E15)_111)'
        dataset_number = 'one_mi((A5_B15_E30)_111)'
        plot_federated_indicator_avg_results(dataset_number)
    elif exp_model == 'Fed_vs_Local':
        dataset_name = '(A5_A10_A15)_532r_One_time'
        data_dir = 'E:/zx/Fed-AQ/experiments_results/f-gan_vs_gan/{}/'.format(dataset_name)
        save_path = 'E:/zx/Fed-AQ/experiments_results/Figures/'
        figure_name = '{}_f_vs_l'.format(dataset_name)
        leg = ['Federated-GAN', 'GAN']
        fs = [f for _, _, f in os.walk(data_dir)]
        fs = fs[0]
        figure_title = ''
        data_files = [data_dir+f for f in fs]
        print(data_files)
        plot_local_indicator_avg_results(data_files, save_path=save_path,
                                         save_name=figure_name, leg=leg, fig_title=figure_title,
                                         file_type='csv')

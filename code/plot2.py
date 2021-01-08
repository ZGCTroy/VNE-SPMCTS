from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', 20)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 400)
pd.set_option('expand_frame_repr', False)


def plot(x, y, filepath, legend, marker, interval=50):
    data = pd.read_csv(filepath)
    data = data.loc[
        data['time periods'] % interval == 0
        ]
    x = data[x]
    y = data[y]
    plt.plot(
        x,
        y,
        label=legend,
        marker=marker
    )


def plot_on_one_iteration(root_dir, policies, y_name, y_lim=None, y_label=None):
    markers = ['^', '.', 'p', '+', 'o', '*']

    plt.figure(dpi=1200)
    plt.style.use("seaborn-muted")
    plt.rc('font', family='Times New Roman', size=13)
    plt.grid(linestyle='--')
    plt.ylabel(y_label)
    plt.xlabel('number of time periods')

    marker_id = 0
    for policy in policies:
        method = '{}_{}_{}_iteration{}_C{}_D{}'.format(
            policy['simulation policy'], policy['expand policy'], policy['reward policy'], policy['iteration'],
            policy['C'], policy['D']
        )
        filepath = os.path.join(
            root_dir,
            method + '.csv'
        )
        plot(
            x='time periods',
            y=y_name,
            filepath=filepath,
            legend=policy['method'],
            marker=markers[marker_id],
            interval=500
        )
        marker_id += 1

    plot(
        x='time periods',
        y=y_name,
        filepath=os.path.join(root_dir, 'VNE-UEPSO.csv'),
        legend='VNE-UEPSO',
        marker=markers[marker_id],
        interval=500
    )

    if y_lim:
        plt.ylim(y_lim)

    plt.legend()

    # plt.xlim(0,20000,5000)
    plt.xticks(np.arange(0,25000,5000))
    if y_name == 'node utilization':
        filename = 'zheng8.eps'
    if y_name == 'acceptance ratio':
        filename = 'zheng9.eps'
    if y_name == 'profitability':
        filename = 'profitability'
    if y_name == 'revenue cost ratio':
        filename = 'zheng10.eps'

    plt.savefig('../figures/' + filename)

    # plt.show()


def plot_on_different_iterations(root_dir, policies, iterations, y_name):
    # plt.figure(dpi=600)
    plt.rc('font', family='Times New Roman', size=13)
    plt.xlabel('number of iterations')
    # plt.ylabel('average physical node utilization')
    plt.ylabel(y_name)

    marker_id = 0
    markers = ['^', '.', 'p', '+']
    legend = []

    for policy in policies:
        x = iterations
        y = []
        for iteration in iterations:
            filename = '{}_{}_{}_iteration{}_C{}_D{}.csv'.format(
                policy['simulation policy'], policy['expand policy'], policy['reward policy'], iteration, policy['C'],
                policy['D']
            )

            filepath = os.path.join(
                root_dir,
                filename
            )

            data = pd.read_csv(filepath)

            if y_name in ['node utilization']:
                y.append(np.average(data[y_name]))
            else:
                y.append(list(data[y_name])[-1])

        plt.plot(
            x,
            y,
            marker=markers[marker_id]
        )
        marker_id += 1
        legend.append(
            policy['method']
        )

    data = pd.read_csv(os.path.join(root_dir, 'VNE-UEPSO.csv'))
    if y_name in ['node utilization']:
        y = np.average(data[y_name])
    else:
        y = list(data[y_name])[-1]
    plt.plot(
        iterations,
        [y for i in iterations],
        marker=markers[marker_id]
    )
    legend.append('VNE-UPESO')

    plt.legend(legend)
    plt.grid(linestyle='--')
    # plt.savefig('../figures/zheng11.eps')
    plt.show()


def plot_one_iteration(root_dir, policies):
    plot_on_one_iteration(
        root_dir=root_dir,
        policies=policies,
        y_name='node utilization',
        y_label='average physical node utilization ratio',
        y_lim=(0.0, 0.75)
    )

    plot_on_one_iteration(
        root_dir=root_dir,
        policies=policies,
        y_name='acceptance ratio',
        y_label='acceptance ratio',
        y_lim=(0.4, 1.05)
    )
    plot_on_one_iteration(
        root_dir=root_dir,
        policies=policies,
        y_name='revenue cost ratio',
        y_label='long-term revenue-to-cost ratio',
        y_lim=(0.4, 0.9)
    )
    # plot_on_one_iteration(
    #     root_dir=root_dir,
    #     policies=policies,
    #     y_name='profitability'
    # )


def plot_different_iterations(root_dir, policies, iterations):
    plot_on_different_iterations(
        policies=policies,
        root_dir=root_dir,
        iterations=iterations,
        x_name='number of iteration',
        y_name='acceptance ratio',
    )
    # #
    plot_on_different_iterations(
        policies=policies,
        root_dir=root_dir,
        iterations=iterations,
        x_name='number of iteration',
        y_name='revenue cost ratio',
    )
    plot_on_different_iterations(
        policies=policies,
        root_dir=root_dir,
        iterations=iterations,
        x_name='number of iteration',
        y_name='profitability',
    )
    plot_on_different_iterations(
        policies=policies,
        root_dir=root_dir,
        iterations=iterations,
        x_name='number of iteration',
        y_name='node utilization',
    )


def bar_on_different_iterations(root_dir, policies, iterations, y_name, y_label=None, y_lim=None):
    plt.figure(dpi=1200)
    plt.style.use("seaborn-muted")

    # plt.rc('font', family='Times New Roman', size=13)
    plt.xlabel('number of iterations')

    if y_label:
        plt.ylabel(y_label)
    else:
        plt.ylabel(y_label)

    legend = []

    total_y = []
    legend = []
    for policy in policies:
        y = []
        for iteration in iterations:
            filename = '{}_{}_{}_iteration{}_C{}_D{}.csv'.format(
                policy['simulation policy'], policy['expand policy'], policy['reward policy'], iteration, policy['C'],
                policy['D']
            )

            filepath = os.path.join(
                root_dir,
                filename
            )

            data = pd.read_csv(filepath)

            if y_name in ['node utilization']:
                y.append(np.average(data[y_name]))
            else:
                y.append(list(data[y_name])[-1])
        total_y.append(y)
        legend.append(policy)

    data = pd.read_csv(os.path.join(root_dir, 'VNE-UEPSO.csv'))
    if y_name in ['node utilization']:
        y = np.average(data[y_name])
    else:
        y = list(data[y_name])[-1]
    total_y.append([y for i in iterations])

    width = 0.25

    # plt.bar(
    #     [j - width for j in range(len(iterations))],
    #     total_y[2],
    #     width=width,
    # )

    plt.bar(
        [j - width for j in range(len(iterations))],
        total_y[0],
        width=width,
        tick_label=['     5', '     15', '     30', '     50', '     100', '     250'],
    )
    # plt.bar(
    #     [j for j in range(len(iterations))],
    #     [0 for j in range(len(iterations))],
    #     width=0,
    #     tick_label=['5', '15', '30', '50', '100', '250'],
    # )

    plt.bar(
        [j  for j in range(len(iterations))],
        total_y[1],
        width=width,
    )

    plt.grid(linestyle='--', axis='y')
    plt.legend(['MaVEn-S','VNE-SPMCTS'])

    if y_lim:
        plt.ylim(y_lim)

    filename = ''
    if y_name == 'node utilization':
        filename = 'zheng11.eps'
    if y_name == 'acceptance ratio':
        filename = 'zheng12.eps'
    if y_name == 'profitability':
        filename = 'profitability'
    if y_name == 'revenue cost ratio':
        filename = 'zheng13.eps'
    plt.savefig('../figures/' + filename)
    # plt.show()


def bar_different_iterations(root_dir, policies, iterations):
    bar_on_different_iterations(
        root_dir=root_dir,
        policies=policies,
        iterations=iterations,
        y_name='acceptance ratio',
        y_label='acceptance ratio',
        y_lim=(0.4, 0.9)
    )
    bar_on_different_iterations(
        root_dir=root_dir,
        policies=policies,
        iterations=iterations,
        y_name='revenue cost ratio',
        y_label='revenue-to-cost ratio',
        y_lim=(0.5, 0.9)
    )

    bar_on_different_iterations(
        root_dir=root_dir,
        policies=policies,
        iterations=iterations,
        y_name='node utilization',
        y_label='average physical node utilization ratio',
        y_lim=(0.2, 0.6)
    )


def main():
    # filepath3 = '../result/result8_26/VNE-UEPSO.csv'

    # TODO 1

    policies = [
        {'method': 'MaVEn-S', 'reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random',
         'iteration': 15, 'C': 0.5,
         'D': 0},
        {'method': 'VNE-SPMCTS', 'reward policy': 'RC', 'simulation policy': 'DBCPU', 'expand policy': 'DBCPU',
         'iteration': 15, 'C': 0.5,
         'D': 0},
    ]

    plot_one_iteration(
        root_dir='../result/result8_26',
        policies=policies
    )

    # TODO 2
    policies = [
        {'method': 'MaVEn-S', 'reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random', 'C': 0.5,
         'D': 0},
        {'method': 'VNE-SPMCTS', 'reward policy': 'RC', 'simulation policy': 'DBCPU', 'expand policy': 'DBCPU',
         'C': 0.5, 'D': 0},
    ]

    bar_different_iterations(
        root_dir='../result/result8_26',
        policies=policies,
        iterations=[5, 15, 30, 50, 100, 250]
    )


if __name__ == '__main__':
    main()

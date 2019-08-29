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


def plot(x, y, filepath, legend, marker, interval = 50):
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
        # marker=marker
    )


def plot_on_one_iteration(root_dir, policies, y):
    markers = ['^', '.', 'p', '+']

    # plt.figure(dpi=600)
    plt.rc('font', family='Times New Roman', size=13)
    plt.grid(linestyle='--')
    plt.ylabel(y)
    plt.xlabel('time periods')

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
            y=y,
            filepath=filepath,
            legend=method,
            marker=markers[0],
            interval = 200
        )
        marker_id += 1

    plot(
        x='time periods',
        y=y,
        filepath=os.path.join(root_dir, 'VNE-UEPSO.csv'),
        legend='VNE-UEPSO',
        marker=markers[0]
    )

    plt.legend()

    # plt.savefig('../figures/zheng9.eps')
    plt.show()


def plot_on_different_iterations(root_dir, policies, iterations, x_name, y_name):
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
            '{}_{}_{}_C{}_D{}'.format(
                policy['reward policy'], policy['simulation policy'], policy['expand policy'], policy['C'], policy['D']
            )
        )

    plt.legend(legend)
    plt.grid(linestyle='--')
    # plt.savefig('../figures/zheng11.eps')
    plt.show()


def plot_one_iteration(root_dir, policies):
    plot_on_one_iteration(
        root_dir=root_dir,
        policies=policies,
        y='node utilization'
    )

    plot_on_one_iteration(
        root_dir=root_dir,
        policies=policies,
        y='acceptance ratio'
    )
    plot_on_one_iteration(
        root_dir=root_dir,
        policies=policies,
        y='revenue cost ratio'
    )
    plot_on_one_iteration(
        root_dir=root_dir,
        policies=policies,
        y='profitability'
    )


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


def main():
    # filepath3 = '../result/result8_26/VNE-UEPSO.csv'

    # TODO 1
    policies = [
        {'method':'MaVEn-S','reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random', 'iteration': 5, 'C': 0.5,
         'D': 0},
        {'method':'MaVEn-S','method':'MaVEn-S','reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random', 'iteration': 15, 'C': 0.5,
         'D': 0},
        {'method':'MaVEn-S','reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random', 'iteration': 30, 'C': 0.5,
         'D': 0},
        {'method':'MaVEn-S','reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random', 'iteration': 50, 'C': 0.5,
         'D': 0},
        {'method':'MaVEn-S','reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random', 'iteration': 100, 'C': 0.5,
         'D': 0},
        {'method':'MaVEn-S','reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random', 'iteration': 250, 'C': 0.5,
         'D': 0},
    ]

    policies = [
        {'method': 'MaVEn-S', 'reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random',
         'iteration': 5, 'C': 0.5,
         'D': 0},
        {'method':'VNE-SPMCTS','reward policy': 'RC', 'simulation policy': 'DBCPU', 'expand policy': 'DBCPU', 'iteration': 5, 'C': 0.5,
         'D': 0},
        {'method': 'MaVEn-S', 'reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random',
         'iteration': 50, 'C': 0.5,
         'D': 0},
        {'method':'VNE-SPMCTS','reward policy': 'RC', 'simulation policy': 'DBCPU', 'expand policy': 'DBCPU', 'iteration': 50, 'C': 0.5,
         'D': 0},

    ]

    plot_one_iteration(
        root_dir='../result/result8_26',
        policies=policies
    )

    # TODO 2
    policies = [
        {'method':'VNE-SPMCTS','reward policy': 'RC', 'simulation policy': 'DBCPU', 'expand policy': 'DBCPU', 'C': 0.5, 'D': 0},
        {'method':'MaVEn-S','reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random', 'C': 0.5, 'D': 0},
    ]

    iterations = [5, 15, 30, 50, 100, 250]

    plot_different_iterations(
        root_dir='../result/result8_26',
        policies=policies,
        iterations=iterations
    )


if __name__ == '__main__':
    main()
    # policies = [
    #     {'simulation policy': 'LAHF', 'expand policy': 'HCPUF'},
    #     {'simulation policy': 'random', 'expand policy': 'random'},
    # ]
    # plot_on_different_iterations(
    #     policies=policies,
    #     filepath='../result8_21/result1-250-3.csv',
    #     x_name='number of iteration',
    #     y_name='link utilization',
    # )

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', 20)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 400)
pd.set_option('expand_frame_repr', False)


def plot(x, y,filepath,legend,marker):
    result = pd.read_csv(filepath)

    plt.plot(
        result[x],
        result[y],
        label=legend,
        # marker=marker
    )

def plot_on_one_iteration(filepath1,filepath2,filepath3,y):

    markers = ['^', '.', 'p','+']

    # time periods CPU
    # # plt.figure(dpi=600)
    plt.rc('font', family='Times New Roman',size=13)
    plt.grid(linestyle='--')
    plt.ylabel(y)
    plt.xlabel('time periods')

    plot(
        x='time periods',
        y=y,
        filepath=filepath1,
        legend='VNE-SPMCTS',
        marker=markers[0]
    )

    plot(
        x='time periods',
        y=y,
        filepath=filepath2,
        legend='MaVEn-S',
        marker=markers[1]
    )

    plot(
        x='time periods',
        y=y,
        filepath=filepath3,
        legend='VNE-UEPSO',
        marker=markers[2]
    )

    plt.legend()

    # plt.savefig('../figures/zheng9.eps')
    plt.show()



def plot_on_different_iterations(filepath,policies,x_name,y_name):
    data = pd.read_csv(filepath)
    # plt.figure(dpi=600)
    plt.rc('font', family='Times New Roman',size=13)
    plt.xlabel('number of iterations')
    # plt.ylabel('average physical node utilization')
    plt.ylabel(y_name)
    i = 0
    markers = ['^','.','p']
    for policy in policies:
        new_data = data.loc[
            data['method'] == policy['method']
        ]

        plt.plot(
            new_data[x_name],
            new_data[y_name],
            marker = markers[i]
        )
        i += 1
    plt.legend(['VNE-SPMCTS','MaVEn-S'])


    plt.grid(linestyle='--')
    # plt.savefig('../figures/zheng11.eps')
    plt.show()


def main():
    iteration = '75'
    filepath1 = '../result/result8_26/random_LAHF_R-C_iteration'+iteration+'_C10000_D10000.csv'
    filepath2 = '../result/result8_26/random_random_R-C_iteration'+iteration+'_C10000_D0.csv'
    filepath3 = '../result/result8_26/VNE-UEPSO.csv'

    plot_on_one_iteration(filepath1, filepath2, filepath3, y='node utilization')
    plot_on_one_iteration(filepath1, filepath2, filepath3, y='acceptance ratio')
    plot_on_one_iteration(filepath1, filepath2, filepath3, y='revenue cost ratio')
    plot_on_one_iteration(filepath1, filepath2, filepath3, y='profitability')

    policies = [
        {'method':'VNE-SPMCTS'},
        {'method':'MaVEn-S'}
    ]
    filepath = '../result/result8_26/result1-250-3.csv'


    plot_on_different_iterations(
        policies=policies,
        filepath=filepath,
        x_name='number of iteration',
        y_name='acceptance ratio',
    )
    # #
    plot_on_different_iterations(
        policies=policies,
        filepath=filepath,
        x_name='number of iteration',
        y_name='revenue cost ratio',
    )
    plot_on_different_iterations(
        policies=policies,
        filepath=filepath,
        x_name='number of iteration',
        y_name='profitability',
    )
    plot_on_different_iterations(
        policies=policies,
        filepath=filepath,
        x_name='number of iteration',
        y_name='node utilization',
    )
    plot_on_different_iterations(
        policies=policies,
        filepath=filepath,
        x_name='number of iteration',
        y_name='link utilization',
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

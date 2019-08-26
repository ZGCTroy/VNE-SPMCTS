from Monte_Carlo_Tree_Search import *
from Read_Data import *
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.width', 1000)  # 设置字符显示宽度
distance = {}
INF = 999999999


def single_network_mapping(network: NetWork, iter_times, simulation_policy,expand_policy, reward_policy, C, D):
    """
    :param SGraph: Graph型，物理网络
    :param VGraph: Graph型，虚拟网络
    :return: mapping, log
    """
    # TODO 1 : 初始状态
    best_mapping = dict(map(lambda x: (x, None), network.VG.nodes.keys()))

    root = Node(
        depth=0,
        mapping=best_mapping,
        parent=None
    )

    # TODO 2 : 寻找最优映射方案
    tree = MonteCarloTree(
        network=network,
        distance=distance,
        simulation_policy = simulation_policy,
        expand_policy = expand_policy,
        reward_policy=reward_policy,
        C=C,
        D=D
    )
    for i in range(network.VG.node_num):
        best_child = tree.UCTSearch(
            root=root,
            iter_times=iter_times
        )
        if best_child == None:
            return None, None

        root = Node(
            parent=None,
            mapping=best_child.mapping,
            depth=best_child.depth
        )

    # TODO 3 : 导入映射方案，判断是否满足链路带宽需求
    best_mapping = root.mapping
    if tree.Evaluate_Reward(best_mapping) == -INF:
        return None, None
    # TODO 4 : 进行链路映射消耗
    network.load_mapping(best_mapping)
    reward, edge_cost, edge_cost_list = network.bandwidth_cost(consumable=True)
    # TODO 4 : 进行节点映射消耗
    cpu_cost, cpu_cost_list = network.cpu_consume(consumable=True)

    # TODO 5 : 返回映射方案mapping 和 本次映射的消耗记录log（用于之后撤销映射的恢复）
    log = Log(
        cpu_cost_list=cpu_cost_list,
        edge_cost_list=edge_cost_list,
        cpu_cost=cpu_cost,
        edge_cost=edge_cost,
        vn_id=network.VG.id
    )
    return best_mapping, log


def receive_sequential_requests(data, iter_times, simulation_policy,expand_policy, reward_policy, C, D):
    # TODO 1 : parameters
    network = NetWork(
        SG=copy.deepcopy(data.SGraphs[0]),
        VG=data.VGraphs[0]
    )
    requests = [i for i in range(500)]
    requests.remove(209)
    time_periods = []
    cpu_utilizations = []
    total_utilizations = []
    revenue_cost_ratios = []
    acceptance_ratios = []
    cpu_stds = []
    logs = {}
    initial_SG_node_sources = list(network.SG.nodes.values())
    initial_SG_cpu_sources = network.SG.sum_cpu_sources()
    initial_SG_edge_sources = network.SG.sum_edge_sources()
    initial_total_sources = initial_SG_cpu_sources + initial_SG_edge_sources
    start_time = time.clock()
    revenue = 0
    cost = 0
    alive = 0
    successful_request_num = 0
    total_request_num = 0
    id = 0
    # TODO 2 :single network mapping
    t = 0
    single_rc = []
    ids = []
    while len(requests) > 0:
        # 撤销
        ask_num = 0
        dead = 0
        if t == 0:
            ask_num = 20
        else:
            if t in logs:
                dead = len(logs[t])
                alive -= dead
                for log in logs[t]:
                    network.SG.recover(log)
                ask_num = 1 + dead * 3

        #print(t, 'alive = ', alive, 'left = ', len(requests),'dead = ',dead,'ask_num = ',ask_num)

        continuous_failed_num = ask_num
        # 接纳
        failed_requests = []
        #print('time = ',t)
        while len(requests) > 0 and continuous_failed_num > 0:
            total_request_num += 1
            request_id = requests.pop(0)  # 队首出队

            network.VG = data.VGraphs[request_id]
            mapping, log = single_network_mapping(
                network=network,
                iter_times=iter_times,
                simulation_policy = simulation_policy,
                expand_policy = expand_policy,
                reward_policy=reward_policy,
                C=C,
                D=D
            )

            if mapping == None:
                continuous_failed_num -= 1
                failed_requests.append(request_id)
            else:

                continuous_failed_num = ask_num
                successful_request_num += 1
                alive += 1
                if t + network.VG.life_time not in logs:
                    logs[t + network.VG.life_time] = [log]
                else:
                    logs[t + network.VG.life_time].append(log)
                cost += log.edge_cost + log.cpu_cost
                revenue += (network.VG.edge_sources + network.VG.cpu_sources)
                id = id +1
                # print('t = {} id = {} VGid = {}, AUR = {}, R/C = {}, link cost = {}, cpu cost = {}, long-term cost = {}, long-term revenue = {}, SN edge resources = {}, SN cpu resources = {},r/c={}'.format(
                #     t,
                #     id,
                #     network.VG.id,
                #     (initial_SG_cpu_sources - network.SG.cpu_sources) / initial_SG_cpu_sources,
                #     revenue / cost,
                #     log.edge_cost/1000,
                #     log.cpu_cost/1000,
                #     cost/1000,
                #     revenue/1000,
                #     network.SG.edge_sources,
                #     network.SG.cpu_sources,
                #     (network.VG.edge_sources + network.VG.cpu_sources)/(log.edge_cost + log.cpu_cost)
                # )
                # )
                #ids.append(id)
                #single_rc.append((network.VG.edge_sources + network.VG.cpu_sources)/(log.edge_cost + log.cpu_cost))


        # 计算节点资源利用
        time_periods.append(t)
        revenue_cost_ratios.append(revenue / cost)
        cpu_utilizations.append((initial_SG_cpu_sources - network.SG.cpu_sources) / initial_SG_cpu_sources)
        total_utilizations.append((initial_total_sources - network.SG.edge_sources - network.SG.cpu_sources) / initial_total_sources)
        acceptance_ratios.append(successful_request_num / total_request_num)
        current_SG_node_sources = np.subtract(initial_SG_node_sources, list(network.SG.nodes.values()))
        current_SG_node_utilization = np.divide(current_SG_node_sources, initial_SG_node_sources)
        cpu_std = np.std(current_SG_node_utilization)
        cpu_stds.append(cpu_std)

        # 将失败队列插回请求队列的原来的位置
        while len(failed_requests):
            # requests.insert(0, failed_requests.pop())
            requests.append(failed_requests.pop())
        print(t, 'alive = ', alive, 'left = ', len(requests), 'dead = ', dead, 'ask_num = ', ask_num)
        print('cpu_utilization = ',cpu_utilizations[-1])
        print('revenue cost ratio = ',revenue_cost_ratios[-1],'\n')
        t += 50

        if t>3000:
            print(requests)
            break


    print('Running time = ', time.clock() - start_time)
    print('Final t = ',t)
    print('Average Acceptance Ratio = ', np.average(acceptance_ratios))
    print('Average CPU cpu_utilization = ', np.average(cpu_utilizations))
    print('Revenue Cost Ratio = ', np.average(revenue_cost_ratios[-1]), '\n')

    result = pd.DataFrame(
        data={
            'time periods': time_periods,
            'acceptance ratio': acceptance_ratios,
            'revenue cost ratio': revenue_cost_ratios,
            'cpu utilization': cpu_utilizations,
            'total utilization': total_utilizations,
            'cpu std': cpu_stds
        },
        columns=['time periods', 'cpu utilization', 'revenue cost ratio', 'total utilization', 'acceptance ratio']
    )
    result.to_csv(
         '../result8_20/{}_{}_{}_iteration{}_C{}_D{}.csv'.format(simulation_policy,expand_policy,reward_policy, iter_times, C, D),
        index=False
    )
    return np.average(cpu_utilizations), revenue_cost_ratios[-1], np.average(cpu_stds), time.clock() - start_time


def Floyd(data, is_print):
    for u in data.SGraphs[0].nodes.keys():
        distance[u] = {}
        for v in data.SGraphs[0].nodes.keys():
            if u in data.SGraphs[0].edges and v in data.SGraphs[0].edges[u]:
                distance[u][v] = 1
            else:
                distance[u][v] = 0

    for k in data.SGraphs[0].nodes.keys():
        for u in data.SGraphs[0].nodes.keys():
            for v in data.SGraphs[0].nodes.keys():
                if u == v:
                    continue
                if distance[u][k] and distance[k][v]:
                    if distance[u][v] == 0:
                        distance[u][v] = distance[u][k] + distance[k][v]
                    else:
                        distance[u][v] = min(distance[u][v], distance[u][k] + distance[k][v])

    if is_print:
        for u in data.SGraphs[0].nodes.keys():
            for v in data.SGraphs[0].nodes.keys():
                print(distance[u][v], end=' ')
            print()

def read_data():
    data = Dataset()
    data.read_SGraphs(
        '../data/data7/maprecord.txt'
    )
    data.read_VGraphs(
        '../data/data7/virtualnetworkTP.txt'
    )
    return data


def main():
    # TODO 1 : read data
    origin_data = read_data()
    virtual_node_ranking_data = read_data()
    virtual_node_ranking_data.virtual_node_ranking()

    # TODO 2 : Floyd
    Floyd(origin_data, is_print=False)

    # TODO 3 : multi test
    # reward_policies = ['R-C','(R-C)*CPU']
    reward_policies = ['R-C']
    policies = [
        {'simulation policy': 'random', 'expand policy': 'random', 'C': 2000, 'D': 0,'data': origin_data},
        {'simulation policy': 'LAHF', 'expand policy': 'HCPUF', 'C': 2000, 'D': 2000,'data': virtual_node_ranking_data},
    ]
    # number_of_iteration = [i for i in range(1,50,3)]
    number_of_iteration = [16]
    result = {
        'C': [],
        'D': [],
        'reward policy': [],
        'simulation policy':[],
        'expand policy': [],
        'number of iteration': [],
        'cpu utilization': [],
        'revenue cost ratio': [],
        'cpu std': []
    }
    for iter_times in number_of_iteration:
        for reward_policy in reward_policies:
            for policy in policies:
                average_ut = []
                average_rc = []
                average_std = []
                print(
                    'simulation policy = {},expand policy = {}, reward policy = {}, C = {}, D = {}, number of iterations = {}'.format(
                        policy['simulation policy'],
                        policy['expand policy'],
                        reward_policy,
                        policy['C'],
                        policy['D'],
                        iter_times
                    )
                )

                for i in range(3):
                    ut, rc, std, _ = receive_sequential_requests(
                        data=policy['data'],
                        iter_times=iter_times,
                        simulation_policy=policy['simulation policy'],
                        expand_policy=policy['expand policy'],
                        reward_policy=reward_policy,
                        C=policy['C'],
                        D=policy['D']
                    )
                    average_ut.append(ut)
                    average_rc.append(rc)
                    average_std.append(std)


                print('cpu utilization = ', np.average(average_ut))
                print('revenue cost ratio = ', np.average(average_rc))
                print('cpu std = ', np.average(average_std), '\n')

                result['C'].append(policy['C'])
                result['D'].append(policy['D'])
                result['number of iteration'].append(iter_times)
                result['reward policy'].append(reward_policy)
                result['simulation policy'].append(policy['simulation policy'])
                result['expand policy'].append(policy['expand policy'])
                result['cpu utilization'].append(np.average(average_ut))
                result['revenue cost ratio'].append(np.average(average_rc))
                result['cpu std'].append(np.average(average_std))

    result = pd.DataFrame(
        data= result,
        columns=['C', 'D', 'number of iteration', 'reward policy', 'simulation policy','expand policy','cpu utilization',
                 'revenue cost ratio', 'cpu std']
    )

    result.to_csv('../result8_20/result1-250-3.csv',index=False)


if __name__ == '__main__':
    main()

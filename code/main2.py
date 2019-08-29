from Monte_Carlo_Tree_Search import *
from Read_Data import *
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import _random

pd.set_option('display.width', 1000)  # 设置字符显示宽度
distance = {}
INF = 999999999


def single_network_mapping(network: NetWork, iter_times, simulation_policy, expand_policy, reward_policy, C, D):
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
        simulation_policy=simulation_policy,
        expand_policy=expand_policy,
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
            mapping=copy.deepcopy(best_child.mapping),
            depth=best_child.depth
        )

    # TODO 3 : 导入映射方案，判断是否满足链路带宽需求
    best_mapping = root.mapping
    if reward_policy == 'R-C' and tree.Evaluate_Reward(best_mapping) == -INF:
        return None, None
    if reward_policy == 'RC' and tree.Evaluate_Reward(best_mapping) == 0:
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


def receive_sequential_requests(data, iter_times, simulation_policy, expand_policy, reward_policy, C, D):
    # TODO 1 : parameters
    network = NetWork(
        SG=copy.deepcopy(data.SGraphs[0]),
        VG=data.VGraphs[0]
    )

    network.SG.print_graph()

    requests = [i for i in range(300)]
    requests.remove(209)

    time_periods = []
    node_utilizations = []
    total_utilizations = []
    link_utilizations = []
    revenue_cost_ratios = []
    acceptance_ratios = []
    logs = {}

    initial_SG_node_sources = list(network.SG.nodes.values())
    initial_SG_cpu_sources = network.SG.sum_cpu_sources()
    initial_SG_edge_sources = network.SG.sum_edge_sources()
    initial_total_sources = initial_SG_cpu_sources + initial_SG_edge_sources
    print(initial_SG_cpu_sources, initial_SG_edge_sources, initial_total_sources)

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
    maximum_continuous_failed_num = 20

    while len(requests) > 0:
        # 撤销
        dead = 0

        if t in logs:
            dead = len(logs[t])
            alive -= dead
            for log in logs[t]:
                network.SG.recover(log)

        # 接纳
        accept_num = 0

        continuous_failed_num = maximum_continuous_failed_num
        failed_requests = []
        while len(requests) > 0 and continuous_failed_num > 0:
            total_request_num += 1
            request_id = requests.pop(0)  # 队首出队

            network.VG = data.VGraphs[request_id]
            mapping, log = single_network_mapping(
                network=network,
                iter_times=iter_times,
                simulation_policy=simulation_policy,
                expand_policy=expand_policy,
                reward_policy=reward_policy,
                C=C,
                D=D
            )

            if mapping == None:
                continuous_failed_num -= 1
                failed_requests.append(request_id)
            else:
                continuous_failed_num = maximum_continuous_failed_num
                accept_num += 1
                alive += 1
                if t + network.VG.life_time not in logs:
                    logs[t + network.VG.life_time] = [log]
                else:
                    logs[t + network.VG.life_time].append(log)
                cost += log.edge_cost + log.cpu_cost
                revenue += (network.VG.edge_sources + network.VG.cpu_sources)
                id = id + 1
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
                # ids.append(id)
                # single_rc.append((network.VG.edge_sources + network.VG.cpu_sources)/(log.edge_cost + log.cpu_cost))

        while len(failed_requests):
            requests.insert(0, failed_requests.pop())

        # 计算节点资源利用
        time_periods.append(t)

        revenue_cost_ratios.append(revenue / cost)

        node_utilizations.append((initial_SG_cpu_sources - network.SG.cpu_sources) / initial_SG_cpu_sources)

        total_utilizations.append(
            (initial_total_sources - network.SG.edge_sources - network.SG.cpu_sources) / initial_total_sources)

        link_utilizations.append(
            (initial_SG_edge_sources - network.SG.edge_sources) / initial_SG_edge_sources
        )

        acceptance_ratios.append(successful_request_num / total_request_num)

        print(t, 'dead = ', dead, 'accept = ', accept_num, 'alive = ', alive, 'left = ', len(requests))
        print('node_utilization = ', node_utilizations[-1])
        print('revenue cost ratio = ', revenue_cost_ratios[-1], '\n')
        t += 50
        maximum_continuous_failed_num += 1
        network.SG.print_graph()

        # if node_utilizations[-1] == 0:
        #     break

    print('Running time = ', time.clock() - start_time)
    print('Final t = ', t)
    print('Average Acceptance Ratio = ', np.average(acceptance_ratios))
    print('Average CPU node_utilization = ', np.average(node_utilizations))
    print('Revenue Cost Ratio = ', np.average(revenue_cost_ratios[-1]), '\n')

    result = pd.DataFrame(
        data={
            'time periods': time_periods,
            'acceptance ratio': acceptance_ratios,
            'revenue cost ratio': revenue_cost_ratios,
            'node utilization': node_utilizations,
            'total utilization': total_utilizations,
            'link utilization': link_utilizations
        },
        columns=['time periods', 'node utilization', 'revenue cost ratio', 'total utilization', 'acceptance ratio',
                 'link utilization']
    )
    result.to_csv(
        '../result8_23/{}_{}_{}_iteration{}_C{}_D{}.csv'.format(simulation_policy, expand_policy, reward_policy,
                                                                iter_times, C, D),
        index=False
    )
    return np.average(node_utilizations), revenue_cost_ratios[-1], np.average(
        total_utilizations), np.average(link_utilizations)


def batch_requests(data, iter_times, simulation_policy, expand_policy, reward_policy, C, D):
    # TODO 1 : parameters
    network = NetWork(
        SG=copy.deepcopy(data.SGraphs[0]),
        VG=data.VGraphs[0]
    )

    network.SG.print_graph()

    requests = data.requests

    time_periods = []
    node_utilizations = []
    total_utilizations = []
    link_utilizations = []
    revenue_cost_ratios = []
    acceptance_ratios = []
    profitabilitys = []
    logs = {}

    initial_SG_node_sources = list(network.SG.nodes.values())
    initial_SG_cpu_sources = network.SG.sum_cpu_sources()
    initial_SG_edge_sources = network.SG.sum_edge_sources()
    initial_total_sources = initial_SG_cpu_sources + initial_SG_edge_sources
    # print(initial_SG_cpu_sources, initial_SG_edge_sources, initial_total_sources)

    start_time = time.clock()
    revenue = 0
    cost = 0
    alive = 0
    accept_num = 0

    # TODO 2 :single network mapping

    t = 0
    i = 0
    left = 2000 *1
    total_request_num = 0
    for request in requests:

        # 撤销
        dead = 0
        if t in logs:
            dead = len(logs[t])
            alive -= dead
            for log in logs[t]:
                network.SG.recover(log)

        # 接纳
        for i in range(1):
            for request_id in request:
                left -= 1
    
                network.VG = data.VGraphs[request_id]
    
                if network.VG.cpu_sources == 1:
                    continue
                else:
                    mapping, log = single_network_mapping(
                        network=network,
                        iter_times=iter_times,
                        simulation_policy=simulation_policy,
                        expand_policy=expand_policy,
                        reward_policy=reward_policy,
                        C=C,
                        D=D
                    )
    
                    total_request_num += 1
                    if mapping:
                        accept_num += 1
                        alive += 1
                        if t + network.VG.life_time not in logs:
                            logs[t + network.VG.life_time] = [log]
                        else:
                            logs[t + network.VG.life_time].append(log)
                        cost += log.edge_cost + log.cpu_cost
                        revenue += (network.VG.edge_sources + network.VG.cpu_sources)

        # 计算节点资源利用
        time_periods.append(t)
        acceptance_ratio = accept_num / total_request_num
        acceptance_ratios.append(acceptance_ratio)
        node_utilizations.append(
            (initial_SG_cpu_sources - network.SG.cpu_sources) / initial_SG_cpu_sources
        )
        link_utilizations.append(
            (initial_SG_edge_sources - network.SG.edge_sources) / initial_SG_edge_sources
        )
        total_utilizations.append(
            (initial_total_sources - network.SG.edge_sources - network.SG.cpu_sources) / initial_total_sources
        )
        revenue_cost_ratio = revenue / cost
        revenue_cost_ratios.append(revenue_cost_ratio)
        profitability = acceptance_ratio * revenue_cost_ratio
        profitabilitys.append(profitability)

        print(t, 'dead = ', dead, 'accept = ', accept_num, 'alive = ', alive, 'left = ', left)
        print('node_utilization = ', node_utilizations[-1])
        print('revenue cost ratio = ', revenue_cost_ratio, )
        print('acceptance ratio = ', acceptance_ratio)
        print('profitability', profitability, '\n')

        t += 50
        if t % 500 == 0:
            result = pd.DataFrame(
                data={
                    'time periods': time_periods,
                    'acceptance ratio': acceptance_ratios,
                    'revenue cost ratio': revenue_cost_ratios,
                    'node utilization': node_utilizations,
                    'total utilization': total_utilizations,
                    'link utilization': link_utilizations,
                    'profitability': profitabilitys
                }
            )
            result.to_csv(
                '../result/result8_26/{}_{}_{}_iteration{}_C{}_D{}.csv'.format(simulation_policy, expand_policy,
                                                                               reward_policy,
                                                                               iter_times, C, D),
                index=False
            )
        # if t == 3500:
        #     break

        network.SG.print_graph()

    return np.average(node_utilizations), np.average(
        link_utilizations), np.average(total_utilizations), revenue_cost_ratios[-1], acceptance_ratios[-1], \
           profitabilitys[-1]


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


def read_data(add_augment=False):
    data = Dataset(add_augment=add_augment)
    data.read_SGraphs('../data/data8/maprecord.txt')
    # data.get_random_SG()

    data.read_VGraphs(
        '../data/data8/virtualnetworkTP.txt'
    )
    data.get_period(
        '../data/data8/Evaluation_networkembedding.txt'
    )
    return data


def main():
    # TODO 1 : read data
    origin_data = read_data(add_augment=True)

    # origin_data.SGraphs[0].print_graph()
    # origin_data.print_VGraphs()

    virtual_node_ranking_data = copy.deepcopy(origin_data)
    virtual_node_ranking_data.virtual_node_ranking()

    # TODO 2 : Floyd
    Floyd(origin_data, is_print=False)

    # TODO 3 : multi test
    policies = [
       #  {'method': 'VNE-SPMCTS', 'reward policy': 'RC', 'simulation policy': 'DBCPU', 'expand policy': 'DBCPU',
      #   'C': 0.5, 'D': 0,
     #    'data': virtual_node_ranking_data},

        {'method': 'MaVEn-S', 'reward policy': 'RC', 'simulation policy': 'random', 'expand policy': 'random',
         'C': 0.5, 'D': 0,
        'data': origin_data},

    ]
    # reward_policies = ['R-C','(R-C)*CPU']

   # number_of_iteration = [5, 15, 30, 50, 100, 150, 250]
    number_of_iteration = [5,15,30]

    result = {
        'method': [],
        'reward policy': [],
        'C': [],
        'D': [],
        'simulation policy': [],
        'expand policy': [],
        'number of iteration': [],
        'node utilization': [],
        'link utilization': [],
        'total utilization': [],
        'acceptance ratio': [],
        'revenue cost ratio': [],
        'profitability': []
    }

    for iter_times in number_of_iteration:

        for policy in policies:
            average_node_utilization = []
            average_rc = []
            average_total_utilization = []
            average_link_utilization = []
            average_ac = []
            average_profitability = []

            print(
                'simulation policy = {},expand policy = {}, reward policy = {}, C = {}, D = {}, number of iterations = {}\n'.format(
                    policy['simulation policy'],
                    policy['expand policy'],
                    policy['reward policy'],
                    policy['C'],
                    policy['D'],
                    iter_times
                )
            )

            for i in range(1):
                node_utilization, link_utilization, total_utilization, rc, ac, profitability = batch_requests(
                    data=policy['data'],
                    iter_times=iter_times,
                    simulation_policy=policy['simulation policy'],
                    expand_policy=policy['expand policy'],
                    reward_policy=policy['reward policy'],
                    C=policy['C'],
                    D=policy['D']
                )
                average_node_utilization.append(node_utilization)
                average_link_utilization.append(link_utilization)
                average_total_utilization.append(total_utilization)
                average_rc.append(rc)
                average_ac.append(ac)
                average_profitability.append(profitability)

            print('node utilization = ', np.average(average_node_utilization))
            print('revenue cost ratio = ', np.average(average_rc))
            print('average acceptance ratio', np.average(average_ac))
            print('profitability', np.average(average_profitability), '\n\n')

            result['method'].append(policy['method'])
            result['C'].append(policy['C'])
            result['D'].append(policy['D'])
            result['number of iteration'].append(iter_times)
            result['reward policy'].append(policy['reward policy'])
            result['simulation policy'].append(policy['simulation policy'])
            result['expand policy'].append(policy['expand policy'])
            result['node utilization'].append(np.average(average_node_utilization))
            result['link utilization'].append(np.average(average_link_utilization))
            result['total utilization'].append(np.average(average_total_utilization))
            result['revenue cost ratio'].append(np.average(average_rc))
            result['acceptance ratio'].append(np.average(average_ac))
            result['profitability'].append(np.average(average_profitability))

            csv_result = pd.DataFrame(
                data=result
            )

            # csv_result.to_csv('../result/result8_26/{}_{}_C{}_D{}.csv'.format(
            #    policy['simulation policy'], policy['expand policy'], policy['C'],policy['D']),
            #    index=False
            # )


if __name__ == '__main__':
    main()
    # for i in range(1000000):
    #     a = np.random.normal(loc=10000,scale=2000)
    #     if a<1000:
    #         print(a)
    # link1_utilization = 1.0
    # link2_utilization = 1.0
    # link3_utilization = 1.0
    #
    # pattern = re.compile(r'.*-L1 Ulti is：-(\d+.\d+)-L2 Ulti is：-(\d+.\d+)-L3 Ulti is：-(\d+.\d+)-*')
    # sequence = []
    # filepath = '../data/data7/tmp.txt'
    # with open(filepath, "r") as f:
    #     line = f.readline()
    #     while line:
    #         utilization = re.findall(pattern, line)[0]
    #         utilization = [float(i) for i in utilization]
    #         # print(utilization)
    #
    #         link_resource = utilization[0] * 1280000 + utilization[1] * 16 * 40000 + utilization[2] * 16 * 160000
    #         link_utilization = link_resource / 4480000
    #         print(link_utilization)
    #         line = f.readline()

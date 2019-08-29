import numpy as np
import copy

INF = 999999999


class Log:
    def __init__(self, cpu_cost_list, edge_cost_list, cpu_cost, edge_cost, vn_id):
        self.cpu_cost_list = cpu_cost_list
        self.edge_cost_list = edge_cost_list
        self.cpu_cost = cpu_cost
        self.edge_cost = edge_cost
        self.vn_id = vn_id


class Graph:

    def __init__(self):
        self.id = -1
        self.nodes = {}
        self.edges = {}
        self.cpu_sources = 0
        self.edge_sources = 0
        self.min_cpu_source = 999999
        self.edge_num = 0
        self.node_num = 0
        self.max_bandwidth = 0.0
        self.max_edge_weight = {}
        self.life_time = 0

    def recover(self, log: Log):
        for cpu_cost in log.cpu_cost_list:
            node = cpu_cost[0]
            consumption = cpu_cost[1]
            self.nodes[node] += consumption
            self.cpu_sources += consumption

        for edge in log.edge_cost_list:
            u = edge[0]
            v = edge[1]
            w = edge[2]
            self.edges[u][v] += w
            self.edges[v][u] += w
            self.edge_sources += w

    def node_ranking(self):
        nodes_resources = dict(
            map(lambda x: (x, (1 + self.nodes[x]) * (1 + np.sum([self.edges[x][i] for i in self.edges[x]]))),
                self.nodes.keys()))
        nodes_resources = dict(sorted(nodes_resources.items(), key=lambda d: d[1], reverse=True))
        nodes_resources = dict(map(lambda x: (x, self.nodes[x]), nodes_resources.keys()))
        self.nodes = nodes_resources

    # 计算nodes当前总量
    def sum_cpu_sources(self):
        return np.sum(list(map(lambda x: x[1], self.nodes.items())))

    # 计算edges当前总量
    def sum_edge_sources(self):
        return np.sum(list(map(lambda x: np.sum(list(x.values())), self.edges.values()))) / 2

    # 最短路径
    def Dijkstra(self, from_node, to_node, bandwidth_needed):

        # TODO 1: 初始化
        inf = 0x3f3f3f3f
        distance = {}
        bandwidth = {}
        prenode = {}
        self.path = []
        for v in self.nodes:
            distance[v] = inf
            bandwidth[v] = inf
            prenode[v] = -1

        # TODO 2: Dijkstra
        distance[from_node] = 0
        u = from_node
        while u != to_node:
            if u in self.edges:
                for v in self.edges[u]:
                    if v in distance and self.edges[u][v] > 0 and self.edges[u][v] >= bandwidth_needed:
                        newdis = distance[u] + 1
                        newbandwidth = bandwidth[u] + self.edges[u][v]  # !!! 在路径长度相等的情况下，优先选择链路带宽总量小的路径
                        if newdis < distance[v] or (newdis == distance[v] and newbandwidth < bandwidth[v]):
                            distance[v] = newdis
                            bandwidth[v] = newbandwidth
                            prenode[v] = u
            del distance[u]
            u = min(distance, key=distance.get)

        # TODO 3: 没有可达路径，返回None
        if distance[to_node] == inf:
            return None, None

        # TODO 4: 生成最短路径
        path = []
        u = to_node
        while u != -1:
            path.append(u)
            u = prenode[u]
        path.reverse()
        edges_consumption = []
        for i in range(1, len(path)):
            edges_consumption.append([path[i - 1], path[i], bandwidth_needed])

        # TODO 5: 返回最短路径值和最短路径消耗
        return distance[to_node], edges_consumption

    # 链路消耗，对最短路径经过的路径，进行消耗更新，所有经过路径减去带宽消耗量bandwidth_needed
    def edges_consume(self, edges_consumption):
        for edge in edges_consumption:
            u = edge[0]
            v = edge[1]
            w = edge[2]
            self.edges[u][v] -= w
            self.edges[v][u] -= w
            self.edge_sources -= w

    # 增加节点，并给节点赋值所需/所含的资源量
    def add_nodes(self, u, cpu_sources):
        self.nodes[u] = cpu_sources
        self.min_cpu_source = min(self.min_cpu_source, cpu_sources)
        if u not in self.edges:
            self.edges[u] = {}
        self.max_edge_weight[u] = 0

    # 增加链路
    def add_edges(self, u, v, w):

        if u not in self.edges:
            self.edges[u] = {}
        self.edges[u][v] = w

        if u not in self.max_edge_weight:
            self.max_edge_weight[u] = w
        else:
            self.max_edge_weight[u] = max(w, self.max_edge_weight[u])
        self.edge_num += 1

    # 打印图，包括节点集合nodes与链路集合edges

    def print_graph(self):
        print("{} 号物理网络:".format(self.id))
        print("nodes: {} 个节点".format(self.node_num))
        print(self.nodes)
        print("edges: {} 条边".format(self.edge_num))
        print(self.edges)
        print()
        # print("life time = {}".format(self.life_time))


class NetWork:
    '''
        SG :        图 , 为 物理网络图, 包括物理网络节点资源集合SG.nodes,物理网络链路资源集合SG.edges
        VG :        图 , 为 虚拟网络图, 包括虚拟网络节点资源集合VG.nodes,虚拟网络链路资源集合VG.edges
        Mapping:    虚拟网络节点与物理网络节点的映射方案,为1*6的list,表示0-5号虚拟网络 映射的 物理网络的节点编号
    '''

    def __init__(self, SG: Graph, VG: Graph):
        self.SG = SG
        self.initial_SG = copy.deepcopy(self.SG)
        self.VG = VG
        self.Mapping = {}

    # 导入映射方案（物理网络和虚拟网络固定，无需改动，因而可以多次重用）
    def load_mapping(self, Mapping):
        self.Mapping = Mapping

    # 判断导入的映射方案 是否 满足节点的cpu资源申请
    def cpu_consume(self, consumable):
        cpu_cost_list = []
        cpu_cost = 0
        for i in self.VG.nodes:
            cpu_cost_list.append([self.Mapping[i], self.VG.nodes[i]])
            cpu_cost += self.VG.nodes[i]
            if consumable:
                self.SG.nodes[self.Mapping[i]] -= self.VG.nodes[i]
                self.SG.cpu_sources -= self.VG.nodes[i]
        return cpu_cost, cpu_cost_list

    def bandwidth_cost(self, consumable):
        # TODO 1 : 将需要消耗的链路装入edges中，并按链路带宽从大到小排序，即优先消耗带宽消耗较大的链路（类比瓶子中先装大石子，再装小石子，再装水）
        edges = []
        for u in self.VG.edges:
            for v in self.VG.edges[u]:
                if self.Mapping[u] >= self.Mapping[v]:
                    continue
                edge = (self.Mapping[u], self.Mapping[v], self.VG.edges[u][v])
                edges.append(edge)
        edges.sort(key=lambda item: item[2], reverse=True)

        # TODO 2 : 寻找物理网络上的最短路径（使跳数最少），并进行链路消耗
        reward = 0
        edge_cost = 0
        edge_cost_lists = []
        for edge in edges:
            dis, edge_cost_list = self.SG.Dijkstra(edge[0], edge[1], edge[2])
            if dis == None:
                self.SG.recover(
                    Log(
                        edge_cost_list=edge_cost_lists,
                        edge_cost=edge_cost,
                        cpu_cost=0,
                        cpu_cost_list=[],
                        vn_id=-1
                    )
                )
                return None, None, None

            self.SG.edges_consume(edge_cost_list)
            edge_cost_lists += edge_cost_list
            edge_cost += (dis * edge[2])
            reward += (edge[2])

        if not consumable:
            self.SG.recover(
                Log(
                    edge_cost_list=edge_cost_lists,
                    edge_cost=edge_cost,
                    cpu_cost=0,
                    cpu_cost_list=[],
                    vn_id=-1
                )
            )
        return reward, edge_cost, edge_cost_lists

    def evaluate_reward(self, consumable, reward_policy):
        reward, edge_cost, edge_cost_list = self.bandwidth_cost(consumable)

        if reward_policy == 'R-C':
            if reward == None:
                return -INF
            else:
                return self.VG.edge_sources - edge_cost

        if reward_policy == 'RC':
            if reward == None:
                return 0.0
            elif edge_cost == 0:
                return 1.0
            else:
                return self.VG.edge_sources / edge_cost

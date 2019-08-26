import random

from NetWork import *

INF = 999999999


# Node
class Node:
    def __init__(self, depth, mapping, parent):
        self.mapping = mapping
        self.depth = depth
        self.parent = parent
        self.children = []
        self.untried_actions = []
        self.Q = 0
        self.N = 0
        self.Q2 = 0


# Monte Carlo Tree
class MonteCarloTree:
    '''
    parms:
        network   : Network类，包含固定的虚拟网络和物理网络的图
        distance  :　distance[u][v] 物理网络两节点的最短跳数
        record_mapping_reward : 保存已经评估过的mapping的reward
        rollout_policy : expand时rollout采用的策略,如random,cpu,degree,distance,distance_bandwidth
        vnodes    : 当前要映射的虚拟网络的节点列表
        nodes_available_for_mapping : 根据剩余cpu进行初步筛选,对每个虚拟网络选出满足ｃcpu资源请求的物理节点列表
    '''

    def __init__(self, network: NetWork, distance, simulation_policy, expand_policy, reward_policy, C, D):
        self.network = network
        self.distance = distance
        self.record_mapping_reward = {}
        self.simulation_policy = simulation_policy
        self.expand_policy = expand_policy
        self.reward_policy = reward_policy
        self.vnodes = list(self.network.VG.nodes.keys())
        self.nodes_available_for_mapping = {}
        self.C = C
        self.D = D

        for VGnode, VGcpu in self.network.VG.nodes.items():
            self.nodes_available_for_mapping[VGnode] = []
            VG_max_edge_weight = self.network.VG.max_edge_weight[VGnode]

            for SGnode, SGcpu in self.network.SG.nodes.items():
                if SGcpu < VGcpu:
                    continue
                SG_max_edge_weight = np.max(list(self.network.SG.edges[SGnode].values()))

                if SG_max_edge_weight < VG_max_edge_weight:
                    continue

                self.nodes_available_for_mapping[VGnode].append(SGnode)

    # 判断树结点是否可以继续拓展
    def check_whether_node_is_expandable(self, node):
        if node.depth == self.network.VG.node_num:
            return False
        if node.untried_actions == []:
            return False
        return True

    # 判断树节点是否是叶子结点
    def check_whether_node_is_terminal(self, node):
        return node.depth == self.network.VG.node_num

    # 奖励策略
    def Evaluate_Reward(self, mapping):
        id = ''
        for i in mapping.values():
            id += str(i) + ','
        if id in self.record_mapping_reward:
            return self.record_mapping_reward[id]
        else:
            self.network.load_mapping(mapping)
            reward = self.network.evaluate_reward(consumable=False, reward_policy=self.reward_policy)
            self.record_mapping_reward[id] = reward
            return reward

    # 遍历节点
    def TreePolicy(self, node):
        root_depth = node.depth
        while not self.check_whether_node_is_terminal(node):
            if self.check_whether_node_is_expandable(node):
                # if node.N < 3 and node.depth>root_depth:
                #     return node
                # else:
                #     return self.Expand(node)
                return self.Expand(node)
            else:
                if node.children != []:
                    node = self.SelectBestChild(node, C=self.C, D=self.D)
                else:
                    return node
        return node

    # rollout选取下一个被expand的子节点
    def RollOut(self, node, policy):

        if policy == 'random':
            return random.choice(node.untried_actions)

        if policy == 'HCPUF':
            cpu_list = list(map(lambda x: (self.network.SG.nodes[x] + 1) * (1 + np.sum(
                [self.network.SG.edges[x][i] for i in self.network.SG.edges[x]])), node.untried_actions))
            # if np.mean(cpu_list) == 0:
            #     cpu_list = list(map(lambda x: self.network.SG.nodes[x], node.untried_actions))
            # cpu_list = np.divide(cpu_list, np.mean(cpu_list))
            # cpu_list = np.power(cpu_list, 3)
            # NR = np.add(cpu_list,1)

            NR = cpu_list
            sigma = np.divide(NR, np.mean(NR))
            sigma = np.power(sigma, 3)
            probabilities = np.divide(sigma, np.sum(sigma))
            return np.random.choice(node.untried_actions, p=probabilities)

        if policy == 'degree':
            vnode = self.vnodes[node.depth]
            needed_edge_weigth = self.network.VG.max_edge_weight[vnode]
            degree_list = []
            for SGnode in node.untried_actions:
                degree = 0
                for SG_edge_weight in self.network.SG.edges[SGnode].values():
                    if SG_edge_weight >= needed_edge_weigth:
                        degree += 1
                degree_list.append(degree)
            degree_list = np.power(5, degree_list)
            total_degree = np.sum(degree_list)
            probabilities = degree_list / total_degree
            return np.random.choice(node.untried_actions, p=probabilities)

        if policy == 'LAHF':
            if node.depth == 0:
                return np.random.choice(node.untried_actions)

            hop = {}

            for SGnode in node.untried_actions:
                hop[SGnode] = 0
                for pre_vnode in self.vnodes:
                    if node.mapping[pre_vnode] == None:
                        break
                    hop[SGnode] += self.distance[SGnode][node.mapping[pre_vnode]]

                hop[SGnode] /= node.depth

            NR = np.max(list(hop.values())) - list(hop.values()) + 1
            sigma = np.divide(NR, np.mean(NR))
            sigma = np.power(sigma, 3)
            probabilities = np.divide(sigma, np.sum(sigma))

            return np.random.choice(
                list(hop.keys()),
                p=probabilities
            )

        if policy == 'distance_bandwidth':
            if node.depth == 0:
                return np.random.choice(node.untried_actions)

            cur_vnode = self.vnodes[node.depth]
            cur_vnode_edges = self.network.VG.edges[cur_vnode]
            distance = {}

            for SGnode in node.untried_actions:
                distance[SGnode] = 0
                for pre_vnode in self.vnodes:
                    if pre_vnode in cur_vnode_edges:
                        pre_snode = node.mapping[pre_vnode]
                        distance[SGnode] += self.distance[SGnode][pre_snode] * cur_vnode_edges[pre_vnode]

            NR = 1 / (list(distance.values()) + 1)
            sigma = np.divide(NR, np.min(NR))
            sigma = np.power(sigma, 1)
            probabilities = np.divide(sigma, np.sum(sigma))

            return np.random.choice(
                list(distance.keys()),
                p=probabilities
            )

    # 拓展
    def Expand(self, node):
        # 根据rollout_policy选取下一个expand的动作choosed_action
        choosed_action = self.RollOut(node, policy=self.expand_policy)

        # 将choosed_action从untried_actions中去除
        node.untried_actions.remove(choosed_action)

        # 执行choosed_action
        child_node = self.Move(node, choosed_action)

        node.children.append(child_node)
        return child_node

    # 执行action,进行状态转移
    def Move(self, node, action):
        new_node = Node(
            depth=node.depth + 1,
            mapping=node.mapping.copy(),
            parent=node
        )
        vnode = self.vnodes[new_node.depth - 1]
        new_node.mapping[vnode] = action
        new_node.untried_actions = self.get_legal_untried_actions(new_node)
        return new_node

    # 计算当前node下可映射的物理节点
    def get_legal_untried_actions(self, node):
        if node.depth == self.network.VG.node_num:
            return []
        vnode = self.vnodes[node.depth]
        untried_actions = self.nodes_available_for_mapping[vnode].copy()

        for snode in node.mapping.values():
            if snode in untried_actions:
                untried_actions.remove(snode)

        return untried_actions

    # 根据UCT选取最优子节点
    def SelectBestChild(self, node, C, D):
        if node.children == []:
            return None
        values = []
        for child in node.children:
            average_Q = child.Q / child.N
            UCB = np.sqrt(2 * np.log(node.N) / child.N)
            variance = np.sqrt(
                (child.Q2 - child.N * average_Q * average_Q + D) / child.N
            )
            if D:
                values.append(average_Q + C * UCB + variance)
            else:
                values.append(average_Q + C * UCB)
        best_child = node.children[np.argmax(values)]
        return best_child

    # 返回更新
    def BackPropagate(self, node, reward):
        while node is not None:
            node.N += 1
            node.Q += reward
            node.Q2 += reward * reward
            node = node.parent

    # 搜索
    def UCTSearch(self, root, iter_times):
        root.untried_actions = self.get_legal_untried_actions(root)
        for i in range(iter_times):
            # print('iteration = {}'.format(i))
            child_node = self.TreePolicy(root)
            # print('non leaf node = ',child_node.mapping)
            reward = self.Simulation(child_node)
            self.BackPropagate(child_node, reward)
            # print('cur simulation reward = {}'.format(reward))
            # print(child_node.mapping)
            # print(child_node.parent.mapping)
            # print('average Q = {}'.format(child_node.parent.Q/child_node.parent.N))
            # print()

        best_child = self.SelectBestChild(root, C=0, D=0)
        return best_child

    # 模拟
    def Simulation(self, node):
        node = copy.deepcopy(node)
        while not self.check_whether_node_is_terminal(node):
            node.untried_actions = self.get_legal_untried_actions(node)
            if node.untried_actions == []:
                return -INF
            choosed_action = self.RollOut(node, policy=self.simulation_policy)
            node = self.Move(node, choosed_action)
        return self.Evaluate_Reward(node.mapping)

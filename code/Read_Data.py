import re
from NetWork import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 显示所有列
pd.set_option('display.max_columns', 20)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 400)
pd.set_option('expand_frame_repr', False)


class Dataset:
    def __init__(self, add_augment=False):
        self.VGraphs = {}
        self.SGraphs = {}
        self.VN_Periods = {}
        self.utilizations = {}
        self.benefit_cost_ratios = {}
        self.time_periods = []
        self.requests = []
        self.add_augment = add_augment

    # 读取物理网络Graph集合
    def read_SGraphs(self, file):
        self.SGraphs = {}
        pattern1 = re.compile(r'This is PS network for virtual network-----(\d*)')  # 匹配有用行
        pattern2 = re.compile(r'(\d+).+?(\d+)')  # 匹配有用行
        pattern3 = re.compile(r'from: (\d+) to: (\d+) bw: (\d*.\d*)')  # 匹配有用行
        with open(file, "r") as f:
            line = f.readline()
            while line:
                # TODO 1：读取SGraph编号
                while line and not re.match(pattern1, line):
                    line = f.readline()
                if not re.match(pattern1, line):
                    break
                id = int(re.findall(pattern1, line)[0])  # 提取数字存为list
                SG = Graph()
                SG.id = id
                # TODO 2 : 读入物理网络的节点资源
                while line and not re.match(pattern2, line):
                    line = f.readline()
                while line and re.match(pattern2, line):
                    num = re.findall(pattern2, line)[0]
                    if self.add_augment:
                        SG.add_nodes(int(num[0]), int(num[1]))
                    else:
                        SG.add_nodes(int(num[0]), int(num[1]))
                    line = f.readline()

                # TODO 3 : 读入物理网络的链路资源
                while line and not re.match("This is virtual network", line) and not re.match(pattern3, line):
                    line = f.readline()
                while line and re.match(pattern3, line):
                    num = re.findall(pattern3, line)[0]
                    if self.add_augment:
                        w = float(num[2])
                        SG.add_edges(int(num[0]), int(num[1]), w)
                        SG.add_edges(int(num[1]), int(num[0]), w)
                    else:
                        SG.add_edges(int(num[0]), int(num[1]), float(num[2]))
                        SG.add_edges(int(num[1]), int(num[0]), float(num[2]))
                    line = f.readline()

                SG.node_num = len(SG.nodes)
                SG.cpu_sources = SG.sum_cpu_sources()
                SG.edge_sources = SG.sum_edge_sources()
                SG.max_bandwidth = np.max(list(SG.max_edge_weight.values()))
                self.SGraphs.update({id: SG})

    def get_random_SG(self, ):
        self.SGraphs = {}
        SG = Graph()
        SG.id = 0
        for u in range(1, 65):
            SG.add_nodes(
                u,
                10000
            )
            SG.add_edges(u, u, 0)
            for v in range(u + 1, 65):
                prob = np.random.random()
                if prob < 0.1:
                    w = 10000
                    SG.add_edges(u, v, int(w))
                    SG.add_edges(v, u, int(w))

        SG.node_num = len(SG.nodes)
        SG.cpu_sources = SG.sum_cpu_sources()
        SG.edge_sources = SG.sum_edge_sources()
        SG.max_bandwidth = np.max(list(SG.max_edge_weight.values()))
        self.SGraphs.update({SG.id: SG})

    # 读取虚拟网络Graph图集
    def read_VGraphs(self, file):
        self.VGraphs = {}
        pattern1 = re.compile(r'This is  virtual network-----(\d*)')  # 匹配有用行
        pattern2 = re.compile(r'(\d+).+?(\d+)')  # 匹配有用行
        pattern3 = re.compile(r'from: (\d+) to: (\d+) bw: (\d*.\d*)')  # 匹配有用行
        pattern4 = re.compile(r'The life time is:--- (\d*)')  # 匹配有用行
        with open(file, "r") as f:
            line = f.readline()
            while line:
                # TODO 1：读取VGraph编号
                while line and not re.match(pattern1, line):
                    line = f.readline()
                if not re.match(pattern1, line):
                    break
                id = int(re.findall(pattern1, line)[0])  # 提取数字存为list
                VG = Graph()
                VG.id = id
                # TODO 2 : 读入虚拟网络的节点资源
                while line and not re.match(pattern2, line):
                    line = f.readline()
                while line and re.match(pattern2, line):
                    num = re.findall(pattern2, line)[0]
                    VG.add_nodes(int(num[0]), int(num[1]))
                    line = f.readline()

                # TODO 3 : 读入虚拟网络的链路资源
                while line and not re.match(pattern1, line) and not re.match(pattern3, line):
                    line = f.readline()
                while not re.match(pattern1, line) and line and re.match(pattern3, line):
                    num = re.findall(pattern3, line)[0]
                    VG.add_edges(int(num[0]), int(num[1]), float(num[2]))
                    VG.add_edges(int(num[1]), int(num[0]), float(num[2]))
                    line = f.readline()

                life_time = 0
                if re.match(pattern4, line):
                    life_time = int(re.findall(pattern4, line)[0])
                   # if life_time > 50:

                    #    life_time = (life_time //50) // 2 * 50

                    line = f.readline()

                VG.id = id
                VG.node_num = len(VG.nodes)
                VG.life_time = life_time
                VG.cpu_sources = VG.sum_cpu_sources()
                VG.edge_sources = VG.sum_edge_sources()
                VG.max_bandwidth = np.max(list(VG.max_edge_weight.values()))
                self.VGraphs.update({id: VG})

    # 读取时序
    def get_period(self, file):

        pattern1 = re.compile(r'.*当前接受的虚拟网络数为.*')  # 匹配有用行
        pattern6 = re.compile(r'')  # 匹配benefit cost ratio
        pattern8 = re.compile(r'.*At time: (\d+)--- Process VNs: (.*)')
        self.requests = []

        t = 0
        time_periods = []
        acceptance_ratios = []
        node_utilizations = []
        link_utilizations = []
        revenue_cost_ratios = []
        methods = []
        profitabilitys = []

        with open(file, "r", encoding='GBK') as f:
            line = f.readline()
            while line:
                if re.match(pattern1, line):
                    time_periods.append(t)
                    methods.append('VNE-UEPSO')
                    acceptance_ratios.append(float(re.findall(r'历史接受率为：-(\d+.\d+)-', line)[0]))
                    node_utilizations.append(float(re.findall(r'Utilization rate is:-(\d+.\d+)-', line)[0]))
                    link_utilizations.append(float(re.findall(r'链路资源利用率为：-(\d+.\d+)$', line)[0]))
                    revenue_cost_ratios.append(float(re.findall(r'.*-benifit-cost ratio is:-(\d*.\d*)-*', line)[0]))
                    profitabilitys.append(acceptance_ratios[-1] * revenue_cost_ratios[-1])
                    t += 50

                elif re.match(pattern8, line):
                    find = re.findall(pattern8, line)[0]
                    request_str = str(find[1])
                    request = request_str.split(";")[:-1]
                    self.requests.append(
                        [int(i) for i in request]
                    )

                line = f.readline()

        result = pd.DataFrame(
            data={
                'time periods': time_periods,
                'acceptance ratio': acceptance_ratios,
                'revenue cost ratio': revenue_cost_ratios,
                'node utilization': node_utilizations,
                'link utilization': link_utilizations,
                'profitability':profitabilitys
            }
        )

        result.to_csv(
            '../result/result8_26/VNE-UEPSO.csv',
            index=False
        )

    def virtual_node_ranking(self):
        for i in self.VGraphs:
            self.VGraphs[i].node_ranking()

    def substrate_node_augmentation(self):
        for i in self.SGraphs:
            self.SGraphs[i].node_augmentation()

    # 打印物理网络Graph集
    def print_SGraphs(self):
        for i in self.SGraphs:
            print('{} 号物理网络:'.format(i))
            self.SGraphs[i].print_graph()
            print()
        print('\n\n\n')

    # 打印虚拟网络Graph集
    def print_VGraphs(self):
        for i in self.VGraphs:
            # print('{} 号虚拟网络:'.format(i))
            self.VGraphs[i].print_graph()
            print()
        print('\n\n\n')

    # 打印网络请求序列
    def print_VN_periods(self):
        for i in self.VN_Periods:
            print('\n When time is {} ,the requests for a virtual network are as follows : '.format(i))
            print(self.VN_Periods[i])
        print('\n\n\n')


def main():
    data = Dataset()
    data.get_period('../data/data8/Evaluation_networkembedding.txt')
    # data.read_SGraphs(
    #     '../data/data6/maprecord.txt'
    # )
    # data.read_VGraphs(
    #     '../data/data6/virtualnetworkTP.txt'
    # )
    #
    # data.virtual_node_ranking()
    #
    # data.print_VGraphs()

    # data.get_period(
    #     '../data/data4/recorder.txt'
    # )


if __name__ == '__main__':
    main()

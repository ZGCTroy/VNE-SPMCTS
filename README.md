# 要点
## 1. Introduction

## 2. Virtual Network Embedding Problem 
### 2.1 Description of VNE Problem
### Preliminary Screening
对当前深度的节点，为了得到下一步可以映射的结点列表，可以根据简单的策略对所有未映射的物理节点初步筛选.
#### Cpu Requirement
    一个物理节点可被当前要映射的虚拟节点映射，至少应满足:

    此物理节点的剩余CPU资源　>= 当前要映射的虚拟节点申请的CPU资源

#### Bandwidth Requirement
    一个物理节点可被当前要映射的虚拟节点映射，至少应满足:

    此物理节点直接相连的链路的剩余带宽的最大值　>= 当前要映射的虚拟节点直接相连的链路的剩余带宽的最大值
### 2.2 Peformance Metrics of VNE Problem
#### 2.2.1 Acceptance Ratio
#### 2.2.2 Revenue Cost Ratio
#### 2.2.3 CPU Utilization
#### 2.2.4 Load Balance

## 3. Formulate VNE problem as a Markov Decision Process
### 3.1 Markov Decision Process 
### 3.2 a finite-MDP for VNE
### 3.3 the exact algorithm for MDP 

## 4. Single-Player Monte Carlo Tree Search
### 4.1 Why to use MCTS / Advantage of MCTS
### 4.2 Selction
### 4.3 Expansion
### 4.4 Simulation
#### 4.4.1 Random/Rollout
> the probability of each node_i = 1 / n , n = len(untried_actions)

    untried_actions = [1, 2, 3, 4, 5]

    probabilities   = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    
    return np.random.chocie(untried_actions, p=probabilities)
#### 4.4.2 CPU
> the probability of each node_i = cpu_source / sum(cpu_source)
    
    untried_actions = [1, 2, 3, 4, 5]

    cpu_source_list = [1000, 1000, 1500, 2500, 4000]

    probabilities   = [0.10, 0.10, 0.15, 0.25, 0.40]
    
#### 4.4.3 Degree
Degree (not so good)
> the probability of each node_i = degree / sum(degree)
    untried_actions = [1, 2, 3, 4, 5]

    degree_list = [1,2,2,2,3]

    probabilities   = [0.10, 0.20, 0.20, 0.20, 0.30]

    return np.random.chocie(untried_actions, p=probabilities)
#### 4.4.4 Distance (which i think may be the best policy) 
> the probability of each node_i = (1/distance) / sum(1/distance)

    We prefer to choose the node 
    that is closer to the nodes we've already mapped in previous steps 
    when mapping a virtual network node into a PS network node
    
    For instance, We've already mapped two nodes in previous steps :
        
        nodes_mapped_in_previous_steps = {
            'Vnode1'   : 'PSnode1',
            'Vnode2'   : 'PSnode2',
            'Vnode3'   :   None
        }
    
    and now we want to map the third node 'Vnode3'
    
    First, We've got a actions list that is filled with nodes 
    which is available to be mapped (satisfying the cpu requirement):
       
        untried_actions = [
            'PSnode3',
            'PSnode4',
            'PSnode5'
        ]
       
    Then we calculate the sum of distances to all nodes mapped in previous steps 
    for those in untried_actions 

    total_distance['PSnode3'] 
    =  distance['PSnode3']['PSnode1']
            +distanc['PSnode3']['PSnode2']
    
    total_distance['PSnode4'] 
    =  distance['PSnode4']['PSnode1']
            +distanc['PSnode4']['PSnode2']
    
    total_distance['PSnode5'] 
    =  distance['PSnode5']['PSnode1']
            +distanc['PSnode5']['PSnode2']
    
    distance[u][v] －－ the minimum distance between PS node u and PS node v,
    should be calculated in Floyd Method (O(n^3)) before Monte Carlo Tree Search
    
    Finally,by sorting the probability according to total_distance and using np.random.chocie(), 
    we are able to choose nodes with smaller total_distance in a higher probability 
#### 4.4.5 Distance * bandwidth
> the probability of each node_i = (1/distance*bandwidth) / sum(1/distance*bandwidth)

    total_distance['PSnode3'] 
    =  distance['PSnode3']['PSnode1'] * bandwidth['PSnode3']['PSnode1']
            +distanc['PSnode3']['PSnode2'] * bandwidth['PSnode3']['PSnode2']
            
### 4.5 BackPropagation
#### 4.5.1 Reward Policy

Mapping failed

    reward = 0
    
    或　
    
    reward = -1

Mapping succeed    
* reward不仅仅与物理网络消耗的链路资源有关,还与虚拟网络本身所请求的链路资源有关
    
    reward = 虚拟网络请求的总链路资源 - 消耗物理网络的总链路资源 
    
* reward除了跟虚拟网络和物理网络消耗的链路资源有关外，还跟映射的物理节点的总CPU剩余量有关

  考虑到Ａ映射方案跟Ｂ映射方案的总链路消耗一致的情况下，是否两种映射方案的reward也一样呢？
  
  在此假设，若消耗的链路资源一致的情况下，优先选择节点CPU资源多的节点，即对映射方案的总剩余CPU进行奖励
  
  reward += alpha * sum(cpu) / 10000 / 虚拟网络节点个数


## 5. Implementation and Experiments Results

### 5.1 Generation of Virtual Network and Substrate Network using ???

### 5.2 MCTS and SP-MCTS
![](figures/MCTS_SP-MCTS_RC.png)

![](figures/MCTS_SP-MCTS_Ut.png)

### 5.3 Different Simulation Policies
![](figures/ut.png)

![](figures/multi_ut.png)

![](figures/multi_ut2.png)

![](figures/bc.png)

![](figures/multi_bc.png)

![](figures/multi_bc2.png)

### 5.4 Different Reward Policies






 
## 3 链路映射
### 3.1 Dijkstra的优先顺序

1. 将需要消耗的虚拟链路装入edges = []

        edge有三个属性,edge[0],edge[1],edge[2]

        在未映射前,edge 三个属性分别表示两个网络上的　一条链路的起始节点，终止节点和链路带宽

        在映射时，edge[0],edge[1]分别为两个虚拟网络节点映射到物理网络上对应的两个物理节点,edge[2]为链路带宽
    
2. edge[1]按链路带宽从大到小排序，即优先消耗带宽消耗较大的链路

        类比瓶子中先装大石子，再装小石子，再装水。


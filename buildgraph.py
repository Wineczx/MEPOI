""" Build the user-agnostic global trajectory flow map from the sequence data """
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_global_POI_checkin_graph(df, exclude_user=None):
    G = nx.DiGraph() #创建一个有向图
    users = list(set(df['user_id'].to_list())) #set去重，转为列表
    if exclude_user in users: users.remove(exclude_user) 
    loop = tqdm(users) #进度条
    for user_id in loop: #每一个用户
        user_df = df[df['user_id'] == user_id] #筛选出该用户的信息

        # Add node (POI)
        for i, row in user_df.iterrows(): #遍历这个用户的每一行数据
            node = row['business_id'] #地点的标识符
            if node not in G.nodes(): #如果图里没有这个节点
                G.add_node(row['business_id'], #增加一个节点，包括经纬度啥的信息
                           checkin_cnt=1,
                           poi_catid=row['cat_id'],
                           poi_catid_code=row['cat_id'],
                           poi_catname=row['cat_id'],
                           latitude=row['latitude'],
                           longitude=row['longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1 #重复经过一个地点

        # Add edges (Check-in seq)
        previous_poi_id = 0 #记录上一个地点
        previous_traj_id = 0 #记录上一个轨迹的id
        for i, row in user_df.iterrows(): #对于用户去的每一个地方
            poi_id = row['business_id'] #当前行的地点
            traj_id = row['user_id']  #当前行的轨迹id
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id): #如果以前没去过或者前一个轨迹与当前轨迹id不同
                previous_poi_id = poi_id #赋值
                previous_traj_id = traj_id #赋值
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id): #如果g已经有了这条轨迹
                G.edges[previous_poi_id, poi_id]['weight'] += 1 #权重加一
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1) #增加一条新的边
            previous_traj_id = traj_id #赋值
            previous_poi_id = poi_id

    return G


def save_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes() #图的节点列表
    A = nx.adjacency_matrix(G, nodelist=nodelist) #生成一个图的邻接矩阵a，notelist决定了节点的顺序
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.savetxt(os.path.join(dst_dir, 'gn3/graph_A.csv'), A.todense(), delimiter=',') #将稀疏矩阵转化为密集矩阵并保存为csv格式文件

    # Save nodes list
    nodes_data = list(G.nodes.data()) #节点数据的列表 # [(node_name, {attr1, attr2}),...]
    with open(os.path.join(dst_dir, 'gn3/graph_X.csv'), 'w') as f: #打开一个csv文件并写到指定目录下
        print('node_name/poi_id,checkin_cnt,poi_catid,poi_catid_code,poi_catname,latitude,longitude', file=f)
        for each in nodes_data: #对于每一个节点的数据
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            poi_catid = each[1]['poi_catid']
            poi_catid_code = each[1]['poi_catid_code']
            poi_catname = each[1]['poi_catname']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            print(f'{node_name},{checkin_cnt},'
                  f'{poi_catid},{poi_catid_code},{poi_catname},'
                  f'{latitude},{longitude}', file=f)


def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'gn3/graph.pkl'), 'wb')) #保存图形数据


def save_graph_edgelist(G, dst_dir): #保存图的边列表
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)} #创建一个字典，将节点列表的每个节点与对应的索引值进行映射

    with open(os.path.join(dst_dir, 'gn3/graph_node_id2idx.txt'), 'w') as f: #写入一个文本文件
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, 'gn3/graph_edge.edgelist'), 'w') as f: #边写入文件中
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")


if __name__ == '__main__':
    dst_dir = r'explainable-recommendation/dataset/PAA'

    # Build POI checkin trajectory graph
    train_df = pd.read_csv(os.path.join(dst_dir, 'train.csv'),delimiter='|')
    print('Build global POI checkin graph -----------------------------------')
    G = build_global_POI_checkin_graph(train_df)

    # Save graph to disk
    save_graph_to_pickle(G, dst_dir=dst_dir)
    save_graph_to_csv(G, dst_dir=dst_dir)
    save_graph_edgelist(G, dst_dir=dst_dir)

import argparse
import json
import os
from transformers import pipeline
import torch
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
def sum_norm_embeddings(embeddings: list, gpu_index):
    # 求和成为一个embedding，形状跟单个cls_embedding一样
    embeddings = torch.stack(embeddings).cuda(2)
    summed_embedding = torch.sum(embeddings, dim=0)
    summed_embedding = summed_embedding.squeeze(0) # 降一维
    # 归一化
    normalized_embedding = F.normalize(summed_embedding, p=2, dim=0) # L2范数，对第0维
    normalized_embedding = normalized_embedding.tolist()
    return normalized_embedding
# 读取模态数据文件
modal1_data = []
with open('/data/yelp/PA/modal_cate_embedding.json', 'r') as file:
    for line in file:
        data = json.loads(line)  # 使用 json.loads 函数将字符串解析为字典
        poi_id = list(data.keys())[0]
        cls_embedding = list(data.values())[0]
        modal1_data.append({'poi_id': poi_id, 'cls_embedding': cls_embedding})

modal1_data = pd.DataFrame(modal1_data)
modal2_data = []
with open('/data/yelp/PA/modal_review_summary_embedding.json', 'r') as file:
    for line in file:
        data = json.loads(line)  # 使用 json.loads 函数将字符串解析为字典
        poi_id = list(data.keys())[0]
        cls_embedding = list(data.values())[0]
        modal2_data.append({'poi_id': poi_id, 'cls_embedding': cls_embedding})

modal2_data = pd.DataFrame(modal2_data)
modal3_data = []
with open('/data/yelp/PA/modal_image_embedding.json', 'r') as file:
    for line in file:
        data = json.loads(line)  # 使用 json.loads 函数将字符串解析为字典
        poi_id = list(data.keys())[0]
        cls_embedding = list(data.values())[0]
        modal3_data.append({'poi_id': poi_id, 'cls_embedding': cls_embedding})

modal3_data = pd.DataFrame(modal3_data)
# 创建一个字典来存储每个business_id对应的模态特征
business_features = []
t = modal1_data['poi_id'].unique()
# 遍历每个business_id，并提取每个模态的特征值
for key in t:
    modal1_features = modal1_data[modal1_data['poi_id']==key]['cls_embedding'].values[0]
    modal2_features = modal2_data[modal2_data['poi_id']==key]['cls_embedding'].values[0]
    modal3_features = modal3_data[modal3_data['poi_id']==key]['cls_embedding'].values[0]
    modal1_features = torch.tensor(modal1_features).cuda(2)
    modal2_features = torch.tensor(modal2_features).cuda(2)
    modal3_features = torch.tensor(modal3_features).cuda(2)

    # 将每个模态的特征值组成一个列表
    modal_features = [modal1_features, modal2_features, modal3_features]
    # 调用sum_norm_embeddings函数融合特征值
    fused_embedding = sum_norm_embeddings(modal_features, gpu_index='2')

    # 将融合后的特征值转换为Python列表
    fused_embedding = fused_embedding

    # 将融合后的特征值保存到字典中
    business_features.append({'business_id': key, 'fuse_embedding': fused_embedding})

# 将DataFrame保存为CSV文件
business_features = pd.DataFrame(business_features)
business_features.to_csv('/data/yelp/PA/fused_features.csv', index=False)

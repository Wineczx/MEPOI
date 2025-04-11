# Author: T_Xu(create), S_Sun(modify)

import argparse
import json
import os
from transformers import pipeline
import torch
import torch.nn.functional as F
import time
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class EncoderModel:
    def __init__(self) -> None:
        pass

    def get_embedding(self, input_text):
        return

class CustomPipeline(EncoderModel):
    def __init__(self, model_name, gpu_index):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map={"": 'cuda:'+gpu_index})
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map={"": 'cuda:'+gpu_index})

    def get_embedding(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        decoder_input_ids = self.tokenizer.encode("", return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            hidden_states = outputs.encoder_last_hidden_state
        cls_embedding = hidden_states[:, 0, :]
        return cls_embedding

class SentenceTransformerModel(EncoderModel):
    def __init__(self, model_name, gpu_index) -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda:'+gpu_index)
        self.model.to(self.device)

    def get_embedding(self, input):
        with torch.no_grad():
            outputs = self.model.encode(input, convert_to_tensor=True, device=self.device)
        return outputs

def sum_norm_embeddings(embeddings: list, gpu_index):
    # 求和成为一个embedding，形状跟单个cls_embedding一样
    embeddings = torch.stack(embeddings).cuda(int(gpu_index))
    summed_embedding = torch.sum(embeddings, dim=0)
    summed_embedding = summed_embedding.squeeze(0) # 降一维
    # 归一化
    normalized_embedding = F.normalize(summed_embedding, p=2, dim=0) # L2范数，对第0维
    normalized_embedding = normalized_embedding.tolist()
    return normalized_embedding

def embed_4_modal_meta(file_path, target_file_path):
    count = 1
    time_start = time.time()
    # 先读取一遍filename记录总行数
    total_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            total_num += 1

    target_file = open(target_file_path, 'w')
    # 临时记录poi_id
    temp_poi_id_set = set()

    # 逐条读取filename
    with open(file_path, 'r') as f:
        for line in f:
            # 解析json
            line = json.loads(line)
            kv = line.popitem()
            poi_id = kv[0]
            des = kv[1]
        
            # embed
            if des is None or des == '':
                # 跳过，最后填充0向量
                continue
            else:
                cls_embedding = encoderModel.get_embedding(des)
            cls_embedding = cls_embedding.tolist()

            # 写入文件
            target_file.write(json.dumps({poi_id: cls_embedding}) + '\n')

            # 记录poi_id
            temp_poi_id_set.add(poi_id)
        
            # 计算剩余时间
            time_end = time.time()
            time_left = (time_end - time_start) / (count + 1) * (total_num - count)
            # 将time_left按照时分秒格式输出
            time_left = time.strftime("%H:%M:%S", time.gmtime(time_left))
            if count % 100 == 0:
                print('count:', count, 'poi_id:', poi_id, 'time_left:', time_left)
            count += 1

    # 填充0向量
    cls_embedding = torch.zeros(768, dtype=torch.float32, device=encoderModel.device).tolist()
    for poi_id in poi_id_set:
        if poi_id not in temp_poi_id_set:
            target_file.write(json.dumps({poi_id: cls_embedding}) + '\n')

    target_file.flush()
    target_file.close()

def embed_4_modal_image(file_path, target_file_path, gpu_index):
    # 用于计算时间
    count = 1
    time_start = time.time()
    # 先读取一遍filename记录总行数
    total_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            total_num += 1

    # 打开目标写入文件
    target_file = open(target_file_path, 'w')
    # 临时记录poi_id
    temp_poi_id_set = set()

    # 逐条读取filename
    with open(file_path, 'r') as f:
        for line in f:
            # 解析json
            line = json.loads(line)
            kv = line.popitem()
            poi_id = kv[0]
            des_list = kv[1]

            embeddings = []
            for des in des_list: # 这里没有做空处理，因为image_des中没有空集合
                cls_embedding = encoderModel.get_embedding(des)
                embeddings.append(cls_embedding)
            # 求和，归一化
            normalized_embedding = sum_norm_embeddings(embeddings, gpu_index)
            
            # 写入文件
            target_file.write(json.dumps({poi_id: normalized_embedding}) + '\n')

            # 记录poi_id
            temp_poi_id_set.add(poi_id)
            
            # 计算剩余时间
            time_end = time.time()
            time_left = (time_end - time_start) / (count + 1) * (total_num - count)
            # 将time_left按照时分秒格式输出
            time_left = time.strftime("%H:%M:%S", time.gmtime(time_left))
            if count % 100 == 0:
                print('count:', count, 'poi_id:', poi_id, 'time_left:', time_left)
            count += 1
            
    # 填充0向量
    cls_embedding = torch.zeros(768, dtype=torch.float32, device=encoderModel.device).tolist()
    for poi_id in poi_id_set:
        if poi_id not in temp_poi_id_set:
            target_file.write(json.dumps({poi_id: cls_embedding}) + '\n')

    target_file.flush()
    target_file.close()

def embed_4_modal_review_summary(file_path, target_file_path, gpu_index):
    # 用于计算时间
    count = 1
    time_start = time.time()
    # 先读取一遍filename记录总行数
    total_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            total_num += 1

    # 打开目标写入文件
    target_file = open(target_file_path, 'w')
    # 临时记录poi_id
    temp_poi_id_set = set()

    # 逐条读取filename
    with open(file_path, 'r') as f:
        for line in f:
            # 解析json
            line = json.loads(line)
            kv = line.popitem() # [key, value]
            poi_id = kv[0] # key
            review_summary_list = kv[1] # value
            
            review_summary_embedding = None
            # 判断list是否为空
            if review_summary_list == []:
                # list为空，跳过，最后填充0向量
                continue
            else:
                embeddings = []
                for review in review_summary_list:
                    cls_embedding = encoderModel.get_embedding(review)
                    embeddings.append(cls_embedding)
                # 求和，归一化
                review_summary_embedding = sum_norm_embeddings(embeddings, gpu_index)
            
            # 写入文件
            target_file.write(json.dumps({poi_id: review_summary_embedding}) + '\n')
            
            # 记录poi_id
            temp_poi_id_set.add(poi_id)
            
            # 计算剩余时间
            time_end = time.time()
            time_left = (time_end - time_start) / (count + 1) * (total_num - count)
            # 将time_left按照时分秒格式输出
            time_left = time.strftime("%H:%M:%S", time.gmtime(time_left))
            if count % 100 == 0:
                print('count:', count, 'poi_id:', poi_id, 'time_left:', time_left)
            count += 1
            
    # 填充0向量
    cls_embedding = torch.zeros(768, dtype=torch.float32, device=encoderModel.device).tolist()
    for poi_id in poi_id_set:
        if poi_id not in temp_poi_id_set:
            target_file.write(json.dumps({poi_id: cls_embedding}) + '\n')

    target_file.flush()
    target_file.close()

def embed_4_modal_review(file_path, target_file_path, gpu_index):
    # 用于计算时间
    count = 1
    time_start = time.time()
    # 先读取一遍filename记录总行数
    total_num = 0
    with open(file_path, 'r') as f:
        for line in f:
            total_num += 1

    # 打开目标写入文件
    target_file = open(target_file_path, 'w')
    # 临时记录poi_id
    temp_poi_id_set = set()

    # 逐条读取filename
    with open(file_path, 'r') as f:
        for line in f:
            # 解析json
            line = json.loads(line)
            kv = line.popitem() # [key, value]
            poi_id = kv[0] # key
            review_list = kv[1] # value
            
            review_embedding = None
            # 判断list是否为空
            if review_list == []:
                # list为空，跳过
                continue
            else:
                embeddings = []
                for review in review_list:
                    cls_embedding = encoderModel.get_embedding(review)
                    embeddings.append(cls_embedding)
                # 求和，归一化
                review_embedding = sum_norm_embeddings(embeddings, gpu_index)
            
            # 写入文件
            target_file.write(json.dumps({poi_id: review_embedding}) + '\n')

            # 记录poi_id
            temp_poi_id_set.add(poi_id)
            
            # 计算剩余时间
            time_end = time.time()
            time_left = (time_end - time_start) / (count + 1) * (total_num - count)
            # 将time_left按照时分秒格式输出
            time_left = time.strftime("%H:%M:%S", time.gmtime(time_left))
            if count % 100 == 0:
                print('count:', count, 'poi_id:', poi_id, 'time_left:', time_left)
            count += 1
            
    # 填充0向量
    cls_embedding = torch.zeros(768, dtype=torch.float32, device=encoderModel.device).tolist()
    for poi_id in poi_id_set:
        if poi_id not in temp_poi_id_set:
            target_file.write(json.dumps({poi_id: cls_embedding}) + '\n')

    target_file.flush()
    target_file.close()

def get_poi_id_set(file_path) -> set:
    meta_file = open(file_path, 'r')

    poi_id_set = set()
    for line in meta_file:
        obj = json.loads(line)
        poi_id_set.add(obj['business_id'])

    return poi_id_set

if __name__ == '__main__':
    # paremeters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/', help='the path of the dataset')
    parser.add_argument('--region', type=str, default='', help='the region name of datasets(e.g. California)')
    parser.add_argument('--gpu_index', type=str, default='2', help='the index of cuda')
    args, _ = parser.parse_known_args()

    parent_path = os.path.join(args.dataset_path, args.region)

    # model_name = "declare-lab/flan-alpaca-xl"
    # encoderModel = CustomPipeline(model_name, args.gpu_index)
    model_name = "all-mpnet-base-v2"
    encoderModel = SentenceTransformerModel(model_name, args.gpu_index)

    # 加载poi_id的set集合，用于填充0向量
    meta_file_path = os.path.join(parent_path, '/data/yelp/PA/PA_business_core.json')
    poi_id_set = get_poi_id_set(meta_file_path)

    # 处理pois_description.json，对其做嵌入
    pois_description_file_path = os.path.join(parent_path, '/data/yelp/PA/PA_categories.json')
    modal_meta_embedding_file_path = os.path.join(parent_path, '/data/yelp/PA/modal_cate_embedding.json')
    embed_4_modal_review_summary(pois_description_file_path, modal_meta_embedding_file_path,args.gpu_index)

    # 处理image_description.json，对其做嵌入
    image_description_file_path = os.path.join(parent_path, '/data/yelp/PA/image2review.json')
    modal_image_embedding_file_path = os.path.join(parent_path, '/data/yelp/PA/modal_image_embedding.json')
    embed_4_modal_image(image_description_file_path, modal_image_embedding_file_path, args.gpu_index)

    # 处理review_summary.json，对其做嵌入
    review_summary_file_path = os.path.join(parent_path, '/data/yelp/PA/review_summary.json')
    modal_review_summary_embedding_file_path = os.path.join(parent_path, '/data/yelp/PA/modal_review_summary_embedding.json')
    embed_4_modal_review_summary(review_summary_file_path, modal_review_summary_embedding_file_path, args.gpu_index)

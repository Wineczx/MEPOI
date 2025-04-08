import logging
from sklearn.preprocessing import OneHotEncoder
import os
import json
import pickle
import pathlib
import zipfile
from pathlib import Path
import transformers
from transformers import GPT2Tokenizer
from torch.optim import AdamW,Adam
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import ast
from dataloader import load_graph_adj_mtx, load_graph_node_features_geo
from model import GCN, NodeAttnMap, UserEmbedding,UserEmbeddings, Time2Vec, CategoryEmbeddings, FuseEmbeddings1, FuseEmbeddings2, TransformerModel,RecReg,SelectAndLoss,MLP,GeoEmbeddings
from param_parser import parameter_parser
from utils import rouge_score, bleu_score, increment_path, calculate_laplacian_matrix, zipdir, ids2tokens,now_time,unique_sentence_percent,increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss,ids2tokens,now_time,ndcg_last_timestep,set_all_seeds
from define_dataset import TrajectoryDatasetTrain,TrajectoryDatasetVal,TrajectoryDatasetTest
import random
args = parameter_parser()
torch.cuda.set_device(args.device)

args.feature1 = 'checkin_cnt'
args.feature2 = 'poi_catid'
args.feature3 = 'latitude'
args.feature4 = 'longitude'
args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

seed=args.seed
set_all_seeds(seed)

save_dir = args.save_dir
model_path = os.path.join(save_dir, 'model.pt')
prediction_path = os.path.join(save_dir, args.outf)
    # Setup logger
for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=os.path.join(args.save_dir, f"log_training.txt"),
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
logging.info(args)
with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
    yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
zipf.close()
# %% ====================== Load data ======================
# Read check-in train data
train_df = pd.read_csv(args.data_train,delimiter='|')
val_df = pd.read_csv(args.data_val,delimiter='|')
test_df = pd.read_csv(args.data_test,delimiter='|')
modal_df = pd.read_csv(args.fused_features)
photo_dict = {}

with open(args.image_description, 'r') as f:
    for line in f:
        json_data = json.loads(line)
        key = list(json_data.keys())[0]
        value = json_data[key]  # 直接获取key对应的完整列表
        photo_dict[key] = value

# Build POI graph (built from train_df)
print('Loading POI graph...')
raw_A = load_graph_adj_mtx(args.data_adj_mtx) #载入全局轨迹图
raw_X,geo_list= load_graph_node_features_geo(args.data_node_feats,
                                    args.feature1,
                                    args.feature2,
                                    args.feature3,
                                    args.feature4
                                    ) #载入特征，改为·多模态,去掉cate
logging.info(
    f"raw_X.shape: {raw_X.shape}; "
    f"Four features: {args.feature1}, {args.feature2}, {args.feature3}")
logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
num_pois = raw_X.shape[0]
unique_values = set(geo_list)  # 将列表转换为集合去除重复项
num_geos = len(unique_values)  # 计算集合中元素的数量

# One-hot encoding poi categories

X = np.zeros((num_pois, 1), dtype=np.float32)
X[:, 0] = raw_X[:, 0]


# Normalization
print('Laplician matrix...')
A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')

# 加载节点特征数据
nodes_df = pd.read_csv(args.data_node_feats, delimiter=',')
# 获取唯一的POI IDs并排序
poi_ids = list(nodes_df['node_name/poi_id'].tolist())
# 为POI创建索引映射，确保每次顺序一致
poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))
idx2poi_id_dict = {idx: poi_id for poi_id, idx in poi_id2idx_dict.items()}
# 假设geo_list已经是按顺序排列好的，这里只要创建一个反向映射即可
geo_id2idx_dict = dict(zip(range(len(geo_list)), geo_list))

# 获取唯一的用户IDs并排序
user_ids = sorted(list(set(train_df['user_id'].astype(str).tolist())))
# 为用户创建索引映射，确保每次顺序一致
user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

# 获取用户轨迹列表，并对用户ID进行排序，以保证顺序一致
traj_list = sorted(list(set(train_df['user_id'].tolist())))

# %% ====================== Define dataloader ======================
print('Prepare dataloader...')
train_dataset = TrajectoryDatasetTrain(train_df,poi_id2idx_dict,modal_df,user_id2idx_dict,photo_dict,geo_id2idx_dict,args)
val_dataset = TrajectoryDatasetVal(val_df,poi_id2idx_dict,modal_df,user_id2idx_dict,photo_dict,geo_id2idx_dict,args)
test_dataset = TrajectoryDatasetTest(test_df,poi_id2idx_dict,modal_df,user_id2idx_dict,photo_dict,geo_id2idx_dict,args)

train_loader = DataLoader(train_dataset,
                            batch_size=args.batch,
                            shuffle=True, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x,
                            generator=torch.Generator().manual_seed(seed))
val_loader = DataLoader(val_dataset,
                        batch_size=args.batch,
                        shuffle=False, drop_last=False,
                        pin_memory=True, num_workers=args.workers,
                        collate_fn=lambda x: x,
                        generator=torch.Generator().manual_seed(seed))
test_loader = DataLoader(test_dataset,
                        batch_size=args.batch,
                        shuffle=False, drop_last=False,
                        pin_memory=True, num_workers=args.workers,
                        collate_fn=lambda x: x,
                        generator=torch.Generator().manual_seed(seed))

# %% ====================== Build Models ======================
# Model1: POI embedding model
if isinstance(X, np.ndarray):
    X = torch.from_numpy(X)
    A = torch.from_numpy(A)
X = X.to(device=args.device, dtype=torch.float)
A = A.to(device=args.device, dtype=torch.float)

args.gcn_nfeat = X.shape[1]
poi_embed_model = GCN(ninput=args.gcn_nfeat,
                        nhid=args.gcn_nhid,
                        noutput=args.poi_embed_dim,
                        dropout=args.gcn_dropout)
print(args.poi_embed_dim)
# Node Attn Model
node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)
# %% Model2: User embedding model, nn.embedding
num_users = len(user_id2idx_dict)
user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)
user_seq_model = UserEmbedding(
                                768,
                                args.transformer_nhead,
                                args.transformer_nhid,
                                args.transformer_nlayers,
                                dropout=args.transformer_dropout)

geo_embed_model = GeoEmbeddings(num_geos, args.geo_embed_dim)

mlp_model2 = MLP(768,256,128) #seq降维
mlp_model3 = MLP(768,256,128) #推荐降维

# %% Model5: Embedding fusion models
# embed_fuse_model1 = FuseEmbeddings1( 128, args.user_embed_dim) #user
embed_fuse_model2 = FuseEmbeddings2(args.poi_embed_dim, 128,args.geo_embed_dim) #poi
# 
# %% Model6: Sequence model
args.seq_input_embed = args.poi_embed_dim + 128 + args.user_embed_dim + 128 +args.geo_embed_dim
seq_model = TransformerModel(num_pois,
                             num_geos,
                                args.seq_input_embed,
                                args.transformer_nhead,
                                args.transformer_nhid,
                                args.transformer_nlayers,
                                dropout=args.transformer_dropout)
max_val_score = -np.inf
# Define overall loss and optimizer
optimizer1 = Adam(params=list(poi_embed_model.parameters()) +
                          list(node_attn_model.parameters()) +
                          list(user_embed_model.parameters()) +
                          list(geo_embed_model.parameters()) +
                          list(mlp_model2.parameters()) +
                          list(mlp_model3.parameters()) +
                        #   list(embed_fuse_model1.parameters()) +
                          list(embed_fuse_model2.parameters()) +
                          list(seq_model.parameters())+
                          list(user_seq_model.parameters()) ,
                  lr=args.lr,
                  weight_decay=args.weight_decay)
lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer1, 'min', verbose=True, factor=args.lr_scheduler_factor)
criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
criterion_geo = nn.CrossEntropyLoss(ignore_index=-1) 
# %% Tool functions for training
def input_traj_to_embeddings(sample, poi_embeddings):
    # Parse sample
    traj_id = sample[0]
    input_seq = [each[0] for each in sample[1]]
    input_seq_fuse = [each[2] for each in sample[1]]
    input_seq_geo = [each[6] for each in sample[1]]
    # User to embedding
    user_id = traj_id
    user_idx = user_id
    input = torch.LongTensor([user_idx]).to(device=args.device)
    user_embedding = user_embed_model(input)
    user_embedding = torch.squeeze(user_embedding).to(device=args.device)
#在这里加上seq_transformer来表示用户，用Id表示用户是没有意义的，每个用户只有一条轨迹
    poi_feature_matrix = []
    poi_feature_matrix2 = []
    for idx in range(len(input_seq)):
        poi_id = input_seq[idx]
        poi_embedding =  torch.tensor(ast.literal_eval(input_seq_fuse[idx].values[0])) .to(device=args.device) # Get embedding and add batch dimension
        poi_feature_matrix.append(poi_embedding)
    poi_feature_matrix2 = torch.stack(poi_feature_matrix, dim=0).to(device=args.device)
    src_mask2 = user_seq_model.generate_square_subsequent_mask(poi_feature_matrix2.unsqueeze(0).size(0))
    user_seq_embedding = user_seq_model(poi_feature_matrix2.unsqueeze(0),src_mask2).to(device=args.device)
    # POI to embedding and fuse embeddings``
    user_seq_embedding2 = mlp_model2(user_seq_embedding)
    input_seq_embed = []
    user_seq_embed = []
    for idx in range(len(input_seq)):
        poi_embedding = poi_embeddings[input_seq[idx]]
        poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)
        input_geo = torch.LongTensor([input_seq_geo[idx]]).to(device=args.device)
        poi_geo = geo_embed_model(input_geo)
        poi_geo = torch.squeeze(poi_geo).to(device=args.device)
        cat_embedding = poi_feature_matrix[idx] #多模态特征
        # Fuse user+poi embeds
        meta_embedding = mlp_model3(cat_embedding)
        fused_embedding2 = embed_fuse_model2(poi_embedding, meta_embedding,poi_geo)

        # Concat time, cat after user+poi
        concat_embedding = torch.cat((user_embedding, fused_embedding2), dim=-1)
        # Save final embed
        input_seq_embed.append(concat_embedding)
    input_seq_embed = torch.stack(input_seq_embed, dim=0)
    input_seq_embed = torch.cat((input_seq_embed,user_seq_embedding2),dim=-1)
    return input_seq_embed,user_seq_embedding,poi_feature_matrix
def adjust_pred_prob_by_graph(y_pred_poi,batch_seq_lens,batch_input_seqs):
    y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
    attn_map = node_attn_model(X, A)

    for i in range(len(batch_seq_lens)):
        traj_i_input = batch_input_seqs[i]  # list of input check-in pois
        for j in range(len(traj_i_input)):
            y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

    return y_pred_poi
poi_embed_model = poi_embed_model.to(device=args.device)
user_embed_model = user_embed_model.to(device=args.device)
node_attn_model = node_attn_model.to(device=args.device)
user_seq_model = user_seq_model.to(device=args.device)
geo_embed_model = geo_embed_model.to(device=args.device)
mlp_model2 = mlp_model2.to(device=args.device)
mlp_model3 = mlp_model3.to(device=args.device)
# embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
seq_model = seq_model.to(device=args.device)
def train(args):
    # %% ====================== Train ======================
    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_ndcg20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_geo_loss_list = []
    train_epochs_text_loss_list = []
    train_epochs_photo_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_ndcg20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_geo_loss_list = []
    val_epochs_text_loss_list = []
    val_epochs_photo_loss_list = []
    endepoch = 0
    best_val_loss = float('inf')
    best_photo_loss = float('inf')
    max_val_score = -np.inf
    # For saving ckpt
    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()
        user_embed_model.train()
        node_attn_model.train()
        geo_embed_model.train()
        embed_fuse_model2.train()
        seq_model.train()
        mlp_model2.train()
        mlp_model3.train()
        user_seq_model.train()
        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_ndcg20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_geo_loss_list = []
        train_batches_text_loss_list = []
        train_batches_photo_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)
            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_user = []
            batch_seq_labels_geo = []
            batch_seq_text = []
            batch_seq_photo = []
            batch_seq_fuse = []
            batch_idseq_user = []
            poi_embeddings = poi_embed_model(X, A)
            # Convert input seq to embeddings
            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                user_id = [each[4] for each in sample[2]]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                label_seq_item = [each[0] for each in sample[2]]
                label_seq_geo = [each[6] for each in sample[2]]
                embed = input_traj_to_embeddings(sample, poi_embeddings)
                input_seq_embed = embed[0] 
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_geo.append(torch.LongTensor(label_seq_geo))
            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)
            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)
            y_pred_poi , y_pred_geo  = seq_model(x, src_mask) #, y_pred_cat
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi,batch_seq_lens,batch_input_seqs)
            optimizer1.zero_grad()
            torch.use_deterministic_algorithms(False)
            loss_poi1 = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            loss_geo1 = criterion_geo(y_pred_geo.transpose(1, 2), y_geo)
            torch.use_deterministic_algorithms(True)
            loss_poi = loss_poi1
            loss_geo = loss_geo1
            # Final loss
            loss = loss_poi + loss_geo
            loss.backward(retain_graph=True)
            optimizer1.step()   
            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            ndcg20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            if loss_poi1 is None or torch.isnan(loss_poi1):
                endepoch = 1
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                ndcg20 += ndcg_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_ndcg20_list.append(ndcg20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_geo_loss_list.append(loss_geo.detach().cpu().numpy())
            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                            f'train_batch_loss:{loss.item():.2f}, '
                            f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                            f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                            f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                            f'train_move_geo_loss:{np.mean(train_batches_geo_loss_list):.2f}\n'
                            f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                            f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                            f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                            f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                            f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                            f'train_move_ndcg20:{np.mean(train_batches_ndcg20_list):.4f}\n'
                            f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                            f'traj_id:{batch[sample_idx][0]}\n'
                            f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                            '=' * 100)
        # train end --------------------------------------------------------------------------------------------------------
        poi_embed_model.eval()
        user_embed_model.eval()
        node_attn_model.eval()
        geo_embed_model.eval()
        user_seq_model.eval()
        mlp_model2.eval()
        mlp_model3.eval()
        # embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        seq_model.eval()
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_ndcg20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_geo_loss_list = []
        val_batches_text_loss_list = []
        val_batches_photo_loss_list = []
        all_tokens_test = []
        all_tokens_predict = []
        all_texts_predict = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        for vb_idx, batch in enumerate(val_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)
            idss_predict = []
            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_geo = []
            batch_seq_user=[]
            batch_seq_text=[]
            batch_seq_photo = []
            batch_seq_fuse = []
            batch_idseq_user =[]
            poi_embeddings = poi_embed_model(X, A)

            # Convert input seq to embeddings处理每个用户
            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                user_id = [each[4] for each in sample[2]]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                text_seq = [each[3] for each in sample[2]]  
                label_seq_item = [each[0] for each in sample[2]]
                label_seq_geo = [each[6] for each in sample[2]]
                embed = input_traj_to_embeddings(sample, poi_embeddings)
                input_seq_embed = embed[0]  
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_geo.append(torch.LongTensor(label_seq_geo))
            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)
            # Feedforward

            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)
            y_pred_poi , y_pred_geo = seq_model(x, src_mask) #, y_pred_cat
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi,batch_seq_lens,batch_input_seqs)
            torch.use_deterministic_algorithms(False)
            loss_poi1 = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            loss_geo1 = criterion_geo(y_pred_geo.transpose(1, 2), y_geo)
            torch.use_deterministic_algorithms(True)
            loss_poi = loss_poi1
            loss_geo = loss_geo1
            # Final loss
            loss = loss_poi+loss_geo
            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            ndcg20 = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                ndcg20 += ndcg_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_ndcg20_list.append(ndcg20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            val_batches_geo_loss_list.append(loss_geo.detach().cpu().numpy())
            # Report validation progress
            if (vb_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                            f'val_batch_loss:{loss.item():.2f}, '
                            f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                            f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                            f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                            f'val_move_geo_loss:{np.mean(val_batches_geo_loss_list):.2f} \n'
                            f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                            f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                            f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                            f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                            f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                            f'val_move_ndcg20:{np.mean(val_batches_ndcg20_list):.4f} \n'
                            f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                            f'traj_id:{batch[sample_idx][0]}\n'
                            f'input_seq:{label_seq_item}\n'
                            # f'label_seq:{batch[sample_idx][2]}\n'
                            f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' +
                            '=' * 100)
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_ndcg20 = np.mean(train_batches_ndcg20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_geo_loss = np.mean(train_batches_geo_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_ndcg20 = np.mean(val_batches_ndcg20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_geo_loss = np.mean(val_batches_geo_loss_list)
        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_geo_loss_list.append(epoch_train_geo_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_ndcg20_list.append(epoch_train_ndcg20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_geo_loss_list.append(epoch_val_geo_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_ndcg20_list.append(epoch_val_ndcg20)
        val_epochs_mrr_list.append(epoch_val_mrr)

        # Monitor loss and score
        monitor_loss = epoch_val_geo_loss + epoch_val_poi_loss
        monitor_score = np.mean(epoch_val_ndcg20 + epoch_val_top20_acc)
        
        # Learning rate schuduler
        lr_scheduler1.step(monitor_loss)
        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                    f"train_loss:{epoch_train_loss:.4f}, "
                    f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                    f"train_geo_loss:{epoch_train_geo_loss:.4f}, "
                    f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                    f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                    f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                    f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                    f"train_mAP20:{epoch_train_mAP20:.4f}, "
                    f"train_ndcg20:{epoch_train_ndcg20:.4f}, "
                    f"train_mrr:{epoch_train_mrr:.4f}\n"
                    f"val_loss: {epoch_val_loss:.4f}, "
                    f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                    f"val_geo_loss: {epoch_val_geo_loss:.4f}, "
                    f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                    f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                    f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                    f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                    f"val_mAP20:{epoch_val_mAP20:.4f}, "
                    f"val_ndcg20:{epoch_val_ndcg20:.4f}, "
                    f"val_mrr:{epoch_val_mrr:.4f}")

        # Save model state dict
        if args.save_weights:
            state_dict = {
                'epoch': epoch,
                'poi_embed_state_dict': poi_embed_model.state_dict(),
                'user_embed_state_dict': user_embed_model.state_dict(),
                'node_attn_state_dict': node_attn_model.state_dict(),
                # 'user_seq_state_dict': user_seq_model.state_dict(),
                'geo_embed_state_dict': geo_embed_model.state_dict(),
                'mlp2_state_dict': mlp_model2.state_dict(),
                'mlp3_state_dict': mlp_model3.state_dict(),
                # 'embed_fuse1_state_dict': embed_fuse_model1.state_dict(),
                'embed_fuse2_state_dict': embed_fuse_model2.state_dict(),
                'seq_model_state_dict': seq_model.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'user_id2idx_dict': user_id2idx_dict,
                'poi_id2idx_dict': poi_id2idx_dict,
                'node_attn_map': node_attn_model(X, A),
                'args': args,
                'epoch_train_metrics': {
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_train_poi_loss': epoch_train_poi_loss,
                    'epoch_train_top1_acc': epoch_train_top1_acc,
                    'epoch_train_top5_acc': epoch_train_top5_acc,
                    'epoch_train_top10_acc': epoch_train_top10_acc,
                    'epoch_train_top20_acc': epoch_train_top20_acc,
                    'epoch_train_mAP20': epoch_train_mAP20,
                    'epoch_train_ndcg20': epoch_train_ndcg20,
                    'epoch_train_mrr': epoch_train_mrr
                },
                'epoch_val_metrics': {
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_poi_loss': epoch_val_poi_loss,
                    'epoch_val_top1_acc': epoch_val_top1_acc,
                    'epoch_val_top5_acc': epoch_val_top5_acc,
                    'epoch_val_top10_acc': epoch_val_top10_acc,
                    'epoch_val_top20_acc': epoch_val_top20_acc,
                    'epoch_val_mAP20': epoch_val_mAP20,
                    'epoch_val_ndcg20': epoch_val_ndcg20,
                    'epoch_val_mrr': epoch_val_mrr,
                }
            }
            model_save_dir = os.path.join(args.save_dir, 'checkpoints')
            if monitor_score >= max_val_score:
                if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    print(state_dict['epoch_val_metrics'], file=f)
                max_val_score = monitor_score
                torch.save(user_seq_model.state_dict(), os.path.join(model_save_dir, 'user_seq_model.state.pt'))

        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_geo_loss_list={[float(f"{each:.4f}") for each in train_epochs_geo_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_ndcg20_list={[float(f"{each:.4f}") for each in train_epochs_ndcg20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            print(f'val_epochs_geo_loss_list={[float(f"{each:.4f}") for each in val_epochs_geo_loss_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_ndcg20_list={[float(f"{each:.4f}") for each in val_epochs_ndcg20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)
def test(args):
    test_epochs_top1_acc_list = []
    test_epochs_top5_acc_list = []
    test_epochs_top10_acc_list = []
    test_epochs_top20_acc_list = []
    test_epochs_mAP20_list = []
    test_epochs_ndcg20_list = []
    test_epochs_mrr_list = []
    test_epochs_loss_list = []
    test_epochs_poi_loss_list = []
    test_epochs_geo_loss_list = []
    test_epochs_text_BLEU1_list =[]
    test_epochs_text_BLEU4_list =[]
    test_epochs_text_loss_list = []
    test_epochs_photo_loss_list = []
    all_tokens_test = []
    all_tokens_predict = []
    all_texts_predict = []
    # test --------------------------------------------------------------------------------------------------------

    poi_embed_model.eval()
    user_embed_model.eval()
    node_attn_model.eval()
    user_seq_model.eval()
    geo_embed_model.eval()
    mlp_model2.eval()
    mlp_model3.eval()
    # embed_fuse_model1.eval()
    embed_fuse_model2.eval()
    seq_model.eval()
    test_batches_top1_acc_list = []
    test_batches_top5_acc_list = []
    test_batches_top10_acc_list = []
    test_batches_top20_acc_list = []
    test_batches_mAP20_list = []
    test_batches_ndcg20_list = []
    test_batches_mrr_list = []
    test_batches_loss_list = []
    test_batches_poi_loss_list = []
    test_batches_geo_loss_list = []
    test_batches_text_loss_list = []
    test_batches_photo_loss_list = []
    test_batches_text_BLEU1_list=[]
    test_batches_text_BLEU4_list=[]   
    src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
    for vb_idx, batch in enumerate(test_loader):
        if len(batch) != args.batch:
            src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)
        idss_predict = []
        # For padding
        batch_input_seqs = []
        batch_seq_lens = []
        batch_seq_embeds = []
        batch_seq_labels_poi = []
        batch_seq_labels_geo = []
        batch_seq_user=[]
        batch_seq_text=[]
        batch_seq_photo=[]
        batch_seq_fuse = []
        batch_idseq_user=[]
        poi_embeddings = poi_embed_model(X, A)

        # Convert input seq to embeddings
        for sample in batch:
            # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
            user_id = [each[4] for each in sample[2]]
            input_seq = [each[0] for each in sample[1]]
            label_seq = [each[0] for each in sample[2]]
            text_seq = [each[3] for each in sample[2]]  
            label_seq_item = [each[0] for each in sample[2]]
            label_seq_geo = [each[6] for each in sample[2]]
            embed = input_traj_to_embeddings(sample, poi_embeddings)
            input_seq_embed = embed[0] 
            batch_seq_embeds.append(input_seq_embed)
            batch_seq_lens.append(len(input_seq))
            batch_input_seqs.append(input_seq)
            batch_seq_labels_poi.append(torch.LongTensor(label_seq))
            batch_seq_labels_geo.append(torch.LongTensor(label_seq_geo))
        # Pad seqs for batch training
        batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
        label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
        label_padded_geo = pad_sequence(batch_seq_labels_geo, batch_first=True, padding_value=-1)
        # Feedforward

        x = batch_padded.to(device=args.device, dtype=torch.float)
        y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
        y_geo = label_padded_geo.to(device=args.device, dtype=torch.long)
        y_pred_poi , y_pred_geo = seq_model(x, src_mask) #, y_pred_cat
        y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi,batch_seq_lens,batch_input_seqs)
        torch.use_deterministic_algorithms(False)
        loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
        loss_geo = criterion_geo(y_pred_geo.transpose(1, 2), y_geo)
        torch.use_deterministic_algorithms(True)
        # Final loss
        loss_poi = loss_poi
        loss_geo = loss_geo

        loss = loss_poi+loss_geo
        # Performance measurement
        top1_acc = 0
        top5_acc = 0
        top10_acc = 0
        top20_acc = 0
        mAP20 = 0
        ndcg20=0
        mrr = 0
        batch_label_pois = y_poi.detach().cpu().numpy()
        batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
        for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
            label_pois = label_pois[:seq_len]  # shape: (seq_len, )
            pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
            top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
            top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
            top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
            top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
            mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
            ndcg20 += ndcg_last_timestep(label_pois, pred_pois, k=20)
            mrr += MRR_metric_last_timestep(label_pois, pred_pois)
        test_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
        test_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
        test_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
        test_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
        test_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
        test_batches_ndcg20_list.append(ndcg20 / len(batch_label_pois))
        test_batches_mrr_list.append(mrr / len(batch_label_pois))
        test_batches_loss_list.append(loss.detach().cpu().numpy())
        test_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
        test_batches_geo_loss_list.append(loss_geo.detach().cpu().numpy())
        # Report validation progress
        if (vb_idx % (args.batch * 5)) == 0:
            sample_idx = 0
            logging.info(f' batch:{vb_idx}, '
                            f'test_batch_loss:{loss.item():.2f}, '
                            f'test_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                            f'test_move_loss:{np.mean(test_batches_loss_list):.2f} \n'
                            f'test_move_poi_loss:{np.mean(test_batches_poi_loss_list):.2f} \n'
                            f'test_move_geo_loss:{np.mean(test_batches_geo_loss_list):.2f} \n'
                            f'test_move_top1_acc:{np.mean(test_batches_top1_acc_list):.4f} \n'
                            f'test_move_top5_acc:{np.mean(test_batches_top5_acc_list):.4f} \n'
                            f'test_move_top10_acc:{np.mean(test_batches_top10_acc_list):.4f} \n' 
                            f'test_move_top20_acc:{np.mean(test_batches_top20_acc_list):.4f} \n'
                            f'test_move_mAP20:{np.mean(test_batches_mAP20_list):.4f} \n'
                            f'test_move_ndcg20:{np.mean(test_batches_ndcg20_list):.4f} \n'
                            f'test_move_MRR:{np.mean(test_batches_mrr_list):.4f} \n'
                            f'traj_id:{batch[sample_idx][0]}\n'
                            f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'+
                            '=' * 100)
    # testid end --------------------------------------------------------------------------------------------------------
    # Calculate epoch metrics
    epoch_test_top1_acc = np.mean(test_batches_top1_acc_list)
    epoch_test_top5_acc = np.mean(test_batches_top5_acc_list)
    epoch_test_top10_acc = np.mean(test_batches_top10_acc_list)
    epoch_test_top20_acc = np.mean(test_batches_top20_acc_list)
    epoch_test_mAP20 = np.mean(test_batches_mAP20_list)
    epoch_test_ndcg20 = np.mean(test_batches_ndcg20_list)
    epoch_test_mrr = np.mean(test_batches_mrr_list)
    epoch_test_loss = np.mean(test_batches_loss_list)
    epoch_test_poi_loss = np.mean(test_batches_poi_loss_list)
    epoch_test_geo_loss = np.mean(test_batches_geo_loss_list)
    # Save metrics to list
    test_epochs_loss_list.append(epoch_test_loss)
    test_epochs_poi_loss_list.append(epoch_test_poi_loss)
    test_epochs_geo_loss_list.append(epoch_test_geo_loss)
    test_epochs_top1_acc_list.append(epoch_test_top1_acc)
    test_epochs_top5_acc_list.append(epoch_test_top5_acc)
    test_epochs_top10_acc_list.append(epoch_test_top10_acc)
    test_epochs_top20_acc_list.append(epoch_test_top20_acc)
    test_epochs_mAP20_list.append(epoch_test_mAP20)
    test_epochs_ndcg20_list.append(epoch_test_ndcg20)
    test_epochs_mrr_list.append(epoch_test_mrr)



    with open(os.path.join(args.save_dir, 'metrics-test.txt'), "w") as f:
        print(f'test_epochs_loss_list={[float(f"{each:.4f}") for each in test_epochs_loss_list]}', file=f)
        print(f'test_epochs_poi_loss_list={[float(f"{each:.4f}") for each in test_epochs_poi_loss_list]}', file=f)
        print(f'test_epochs_geo_loss_list={[float(f"{each:.4f}") for each in test_epochs_geo_loss_list]}', file=f)
        print(f'test_epochs_top1_acc_list={[float(f"{each:.4f}") for each in test_epochs_top1_acc_list]}', file=f)
        print(f'test_epochs_top5_acc_list={[float(f"{each:.4f}") for each in test_epochs_top5_acc_list]}', file=f)
        print(f'test_epochs_top10_acc_list={[float(f"{each:.4f}") for each in test_epochs_top10_acc_list]}', file=f)
        print(f'test_epochs_top20_acc_list={[float(f"{each:.4f}") for each in test_epochs_top20_acc_list]}', file=f)
        print(f'test_epochs_mAP20_list={[float(f"{each:.4f}") for each in test_epochs_mAP20_list]}', file=f)
        print(f'test_epochs_ndcg20_acc_list={[float(f"{each:.4f}") for each in test_epochs_ndcg20_list]}', file=f)
        print(f'test_epochs_mrr_list={[float(f"{each:.4f}") for each in test_epochs_mrr_list]}', file=f)

train(args)
model_save_dir = os.path.join(args.save_dir, 'checkpoints')
state_dict = torch.load(rf"{model_save_dir}/best_epoch.state.pt")
state_dict2 = torch.load(os.path.join(model_save_dir, 'user_seq_model.state.pt'))
user_seq_model.load_state_dict(state_dict2)
poi_embed_model.load_state_dict(state_dict['poi_embed_state_dict'])
user_embed_model.load_state_dict(state_dict['user_embed_state_dict'])
node_attn_model.load_state_dict(state_dict['node_attn_state_dict'])
geo_embed_model.load_state_dict(state_dict['geo_embed_state_dict'])
mlp_model2.load_state_dict(state_dict['mlp2_state_dict'])
mlp_model3.load_state_dict(state_dict['mlp3_state_dict'])
# embed_fuse_model1.load_state_dict(state_dict['embed_fuse1_state_dict'])
embed_fuse_model2.load_state_dict(state_dict['embed_fuse2_state_dict'])
seq_model.load_state_dict(state_dict['seq_model_state_dict'])
test(args)

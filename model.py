import math
from transformers import GPT2LMHeadModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter



class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False): #输入特征的维度，隐藏层的维度，不使用掩码
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask 
        self.out_features = nhid #输出特征的维度等于隐藏层的维度
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid))) #定义一个可训练的参数w
        nn.init.xavier_uniform_(self.W.data, gain=1.414) #权重矩阵初始化，使用xaiver初始化方法
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1))) #可训练的张量
        nn.init.xavier_uniform_(self.a.data, gain=1.414) #初始化
        self.leakyrelu = nn.LeakyReLU(0.2) #创建一个relu激活函数，0.2是斜率的负值，用于引入非线性

    def forward(self, X, A): #前向传递的函数，输入特征矩阵x,邻接矩阵a
        Wh = torch.mm(X, self.W) #输入与权重相乘

        e = self._prepare_attentional_mechanism_input(Wh) #转换为注意力机制的输入矩阵e

        if self.use_mask: #如果使用掩码
            e = torch.where(A > 0, e, torch.zeros_like(e))  # 系数设置为0，应用掩码，可以避免节点间不存在边的影响

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) #与self.a的前几行相乘
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :]) #后几行
        e = Wh1 + Wh2.T #相加
        return self.leakyrelu(e) #激活后返回


class GraphConvolution(nn.Module): #图卷积类
    def __init__(self, in_features, out_features, bias=True): #初始化
        super(GraphConvolution, self).__init__()
        self.in_features = in_features #输入特征
        self.out_features = out_features #输出特征
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) #权重
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  #偏置的初始化方式
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self): #重置参数
        stdv = 1. / math.sqrt(self.weight.size(1)) #权重和参数随机初始化
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj): #前向传播函数
        support = torch.mm(input, self.weight) #支持度矩阵
        output = torch.spmm(adj, support) #卷积后的特征
        if self.bias is not None: #如果有偏置就加入到输出中
            return output + self.bias
        else:
            return output

    def __repr__(self): #表示函数
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class GeoEmbeddings(nn.Module):
    def __init__(self, num_pois, embedding_dim):
        super(GeoEmbeddings, self).__init__()

        self.geo_embedding = nn.Embedding(
            num_embeddings=num_pois,
            embedding_dim=embedding_dim,
        )

    def forward(self, poi_idx):
        embed = self.geo_embedding(poi_idx)
        return embed
class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed
# class UserEmbedding(nn.Module):
#     def __init__(self, num_users, hidden_size=768, max_len=100):
#         super(UserEmbedding, self).__init__()
        
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=hidden_size,
#                 nhead=1,
#                 dim_feedforward=hidden_size
#             ),
#             num_layers=1
#         )
        
#         self.user_transformer_embedding = nn.Embedding(num_users, hidden_size)
#         self.input_linear = nn.Linear(768, hidden_size)
#         self.init_pos_encoding(max_len + 1, hidden_size)  # 注意：位置编码长度加1，为CLS预留位置
        

#     def init_pos_encoding(self, max_len, hidden_size):
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(np.log(10000.0) / hidden_size))
#         pos_enc = torch.zeros((1, max_len, hidden_size))
#         pos_enc[0, :, 0::2] = torch.sin(position.float() * div_term)
#         pos_enc[0, :, 1::2] = torch.cos(position.float() * div_term)
#         self.register_buffer('pos_enc', pos_enc)
#     def forward(self, user_ids, poi_features):
#         # 使用user_ids从user_transformer_embedding获取每个用户的cls_token
#         cls_tokens = self.user_transformer_embedding(user_ids).unsqueeze(1)  # 形状为 [batch_size, 1, hidden_size]
        
#         # 对poi_features进行转换，使其形状为 [batch_size, seq_len, hidden_size]
#         poi_features_transformed = self.input_linear(poi_features)
#         poi_features_transformed = poi_features_transformed.unsqueeze(0)
        
#         # 将cls_tokens添加到转换后的poi_features序列的前面
#         poi_features_combined = torch.cat((cls_tokens, poi_features_transformed), dim=1)
        
#         # 使用位置编码
#         if self.pos_enc is not None:
#             seq_len_plus_cls = poi_features_combined.size(1)
#             pos_enc = self.pos_enc[:, :seq_len_plus_cls, :]
#             poi_features_combined += pos_enc.to(poi_features_combined.device)
        
#         # 将组合后的序列输入到Transformer编码器
#         transformer_output = self.transformer(poi_features_combined)
        
#         # 提取第一个编码输出，即cls_token的输出，并返回该cls_output
#         cls_output = transformer_output[:, 0, :].unsqueeze(0)
        
#         return cls_output
#     def train_user_embed(self, user_ids, poi_features):
#         embed_output = self.forward(user_ids, poi_features)
#         self.user_transformer_embedding.weight.data[user_ids, :] = embed_output.detach()
#         return embed_output

#     def get_user_embedding(self, user_ids):
#         user_embed = self.user_transformer_embedding(user_ids)
#         return user_embed
# class UserEmbedding(nn.Module):
#     def __init__(self, num_users, hidden_size=768):
#         super(UserEmbedding, self).__init__()
#         self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
#         self.user_embedding = nn.Embedding(num_users, hidden_size)

#     def forward(self, user_ids, poi_features):
#         # 使用用户ID从user_embedding获取用户嵌入
#         user_embed = self.user_embedding(user_ids)  # [batch_size, hidden_size]
#         poi_features = poi_features.unsqueeze(0)
#         # 输入POI特征到GRU
#         rnn_output, hidden = self.rnn(poi_features)  # rnn_output: [batch_size, seq_len, hidden_size], hidden: [1, batch_size, hidden_size]

#         # 取最后一个时间步的RNN输出
#         rnn_output_last = rnn_output[:, -1, :]  # [batch_size, hidden_size]

#         # 结合用户嵌入和RNN的最后输出
#         combined_output = rnn_output_last + user_embed  # [batch_size, hidden_size]

#         return combined_output

#     def train_user_embed(self, user_ids, poi_features):
#         embed_output = self.forward(user_ids, poi_features)
#         self.user_embedding.weight.data[user_ids, :] = embed_output.detach()
#         return embed_output

#     def get_user_embedding(self, user_ids):
#         user_embed = self.user_embedding(user_ids)
#         return user_embed
class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings1(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings1, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x
class FuseEmbeddings2(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim, third_embed_dim):
        super(FuseEmbeddings2, self).__init__()

        # 更新：结合三个嵌入维度
        embed_dim = user_embed_dim + poi_embed_dim + third_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed, third_embed):
        # 更新：同时将三个嵌入向量合并
        x = self.fuse_embed(torch.cat((user_embed, poi_embed, third_embed), -1))
        x = self.leaky_relu(x)
        return x

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi,num_geo, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        #self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_geo = nn.Linear(embed_size, num_geo)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
       # out_time = self.decoder_time(x)
        out_geo = self.decoder_geo(x)
        return out_poi , out_geo#, out_cat




#可解释生成
class UIPromptWithReg:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser,nitem, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = True
        model.init_prompt(nuser,nitem)
        return model

    def init_prompt(self, nuser,nitem):
        self.src_len = 4
        emsize = self.transformer.wte.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.mlp_user = MLP(emsize, 768, 768)  # Define your hidden_dim and output_dim
        self.mlp_item = MLP(emsize, 768, 768)
        # self.mlp_xu = MLP(640, 768, 768)
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

    # def forward(self, user,user_seq,xu,item,item_emb,text, mask, ignore_index=-100):
    def forward(self, user,user_seq,item,item_emb,text, mask, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)

        # embeddings
       # u_src = user  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        u_src = self.user_embeddings(user)  # Apply MLP to user
        u_src2 = self.mlp_user(user_seq)
        # i_src = self.mlp_xu(xu)  # Apply MLP to item
        i_src2 = self.mlp_item(item_emb)
        src = torch.cat([u_src.unsqueeze(1), u_src2.unsqueeze(1),i_src.unsqueeze(1), i_src2.unsqueeze(1),w_src], 1)  # (batch_size, total_len, emsize)
        # src = torch.cat([i_src.unsqueeze(1),u_src.unsqueeze(1),u_src2.unsqueeze(1) ,w_src], 1)  
        # src = torch.cat([u_src.unsqueeze(1),i_src.unsqueeze(1),w_src], 1) 

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)


class RecReg(UIPromptWithReg, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

class UserEmbedding(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(UserEmbedding, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, src,src_mask):
        # Scale src and apply positional encodin
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        # Apply Transformer encoding with mask, should output shape [sequence_length, embed_size]
        x = self.transformer_encoder(src, src_mask)
        return x.squeeze(0)
#图像选择
from sentence_transformers import SentenceTransformer
from itertools import chain
# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MLP, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.relu(self.fc(x))
#         return out
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
class SelectAndLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim,nuser,args):
        super(SelectAndLoss, self).__init__()
        self.mlp = MLP(input_dim,hidden_dim, output_dim)
        emsize = 768  # 768
        self.user = nn.Embedding(nuser, emsize)
        # 实例化 SentenceTransformer 模型
        model_name = "all-mpnet-base-v2"
        self.encoderModel = SentenceTransformer(model_name).to(args.device)
        # 冻结预训练模型的参数
        for param in self.encoderModel.parameters():
            param.requires_grad = False
        initrange = 0.1
        self.user.weight.data.uniform_(-initrange, initrange)

    def forward(self, user_embedding, candidate_texts, target_text):
        def cosine_similarity(a, b, dim=-1):
            a_n = F.normalize(a, p=2, dim=dim)
            b_n = F.normalize(b, p=2, dim=dim)
            return (a_n * b_n).sum(dim=dim)
        # user_embedding_transformed = self.mlp(user_embedding)
        user_embedding_transformed = self.mlp(user_embedding)

        # 存储每一批中的所有候选嵌入
        batch_candidate_embeddings = []
        for candidate_list in candidate_texts:
            candidate_embeddings = torch.tensor(self.encoderModel.encode(candidate_list, show_progress_bar=False)).to(user_embedding.device)

            batch_candidate_embeddings.append(candidate_embeddings)

        target_embedding = torch.tensor(self.encoderModel.encode(target_text, show_progress_bar=False)).to(user_embedding.device)
        # 存储在每一批次的候选列表中找到的最佳候选项
        selected_text_per_batch = []
        selected_embed_per_batch = []

        # 为每个批次寻找最相似的候选项
        for us_emb, candid_emb, candid_text in zip(user_embedding_transformed, batch_candidate_embeddings, candidate_texts):
            # 计算当前批次的嵌入与候选嵌入的余弦相似度
            sim = cosine_similarity(us_emb[None, :], candid_emb, dim=-1)
            
            selected_idx = sim.argmax().item()

            selected_text_per_batch.append(candid_text[selected_idx])
            selected_embed_per_batch.append(candid_emb[selected_idx])

        selected_embeddings = torch.stack(selected_embed_per_batch)
        selected_text = selected_text_per_batch

        # loss = 1 - cosine_similarity(selected_embeddings, target_embedding)
        # loss = loss.mean()
        
        target_embedding = target_embedding.squeeze(0)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(selected_embeddings, target_embedding)
    # 计算负余弦相似度损失
        loss = 1 - cos_sim.mean()
        return  selected_text,loss

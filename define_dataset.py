from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
    # %% ====================== Define Dataset ======================
bos = '<bos>'
eos = '<eos>'
class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df,poi_id2idx_dict,modal_df,user_id2idx_dict,photo_dict,geo_id2idx_dict,args):
            self.df = train_df
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []
            print(geo_id2idx_dict)
            for traj_id in tqdm(sorted(set(train_df['user_id'].tolist()))):
                traj_df = train_df[train_df['user_id'] == traj_id]
                poi_ids = traj_df['business_id'].to_list()
                text_seq = traj_df['summary'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
                time_feature = traj_df[args.time_feature].to_list()

                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i],modal_df[modal_df['business_id']==poi_ids[i]]['fuse_embedding'],'{} {} {}'.format(bos,text_seq[i], eos),user_id2idx_dict[traj_id],photo_dict[poi_ids[i]],geo_id2idx_dict[poi_idxs[i]]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1],modal_df[modal_df['business_id']==poi_ids[i+1]]['fuse_embedding'],'{} {} {}'.format(bos,text_seq[i+1], eos),user_id2idx_dict[traj_id],photo_dict[poi_ids[i+1]],geo_id2idx_dict[poi_idxs[i+1]]))
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.traj_seqs.append(user_id2idx_dict[traj_id])
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

class TrajectoryDatasetVal(Dataset):
        def __init__(self, df,poi_id2idx_dict,modal_df,user_id2idx_dict,photo_dict,geo_id2idx_dict,args):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []
            for traj_id in tqdm(sorted(set(df['user_id'].tolist()))):
                user_id = traj_id
                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                     continue

                # Ger POIs idx in this trajectory
                traj_df = df[df['user_id'] == traj_id]
                poi_ids = traj_df['business_id'].to_list()
                text_seq = traj_df['summary'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()

                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i],modal_df[modal_df['business_id']==poi_ids[i]]['fuse_embedding'],'{} {} {}'.format(bos,text_seq[i], eos),user_id2idx_dict[traj_id],photo_dict[poi_ids[i]],geo_id2idx_dict[poi_idxs[i]]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1],modal_df[modal_df['business_id']==poi_ids[i+1]]['fuse_embedding'],'{} {} {}'.format(bos,text_seq[i+1], eos),user_id2idx_dict[traj_id],photo_dict[poi_ids[i+1]],geo_id2idx_dict[poi_idxs[i+1]]))
                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(user_id2idx_dict[traj_id])

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])
        
class TrajectoryDatasetTest(Dataset):
        def __init__(self, df,poi_id2idx_dict,modal_df,user_id2idx_dict,photo_dict,geo_id2idx_dict,args):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []
            for traj_id in tqdm(sorted(set(df['user_id'].tolist()))):
                user_id = traj_id
                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                     continue

                # Ger POIs idx in this trajectory
                traj_df = df[df['user_id'] == traj_id]
                poi_ids = traj_df['business_id'].to_list()
                text_seq = traj_df['summary'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()

                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i],modal_df[modal_df['business_id']==poi_ids[i]]['fuse_embedding'],'{} {} {}'.format(bos,text_seq[i], eos),user_id2idx_dict[traj_id],photo_dict[poi_ids[i]],geo_id2idx_dict[poi_idxs[i]]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1],modal_df[modal_df['business_id']==poi_ids[i+1]]['fuse_embedding'],'{} {} {}'.format(bos,text_seq[i+1], eos),user_id2idx_dict[traj_id],photo_dict[poi_ids[i+1]],geo_id2idx_dict[poi_idxs[i+1]]))
                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(user_id2idx_dict[traj_id])

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])


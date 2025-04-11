import numpy as np
import pandas as pd
import json
import geohash
def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A



def load_graph_node_features(path, path2, feature='fuse_embedding'):
    df = pd.read_csv(path, delimiter=',')
    modal_df = pd.read_csv(path2)
    left_on_col = df.columns[0]
    right_on_col = modal_df.columns[0]

    df = pd.merge(df, modal_df, left_on=left_on_col, right_on=right_on_col, how='left')
    
    # 这里假设每个单元格是一个如 "[0.1, 0.2, ..., 0.768]" 的字符串
    # 我们需要将其转换为实际的列表
    features_series = df[feature].apply(lambda x: json.loads(x))

    # 现在每个单元格应该是一个数字列表
    # 我们可以将这些列表堆叠成一个NumPy数组
    X = np.array(features_series.tolist())

    print(X.shape)  # 查看最终数组的形状
    return X
import pandas as pd

def geohash_encode(lat, lon, precision=6):
    """Encode latitude and longitude into geohash."""
    return geohash.encode(lat, lon, precision=precision)

# def load_graph_node_features_geo(path, feature1='checkin_cnt', 
#                                  feature2='latitude', feature3='longitude'):
#     """Load features and encode geo information to geohash, then apply one-hot encoding."""
#     # Load data
#     df = pd.read_csv(path, delimiter='|') 
#     # Assume geohash_encode function is properly defined elsewhere
    
#     # Apply geohash encoding
#     df['geohash'] = df.apply(lambda row: geohash_encode(row[feature2], row[feature3]), axis=1)

#     # Apply one-hot encoding to the geohash
#     geohash_dummies = pd.get_dummies(df['geohash'])
    
#     # Concatenate the checkin_cnt feature with the geohash dummy variables
#     X = pd.concat([df[[feature1]], geohash_dummies], axis=1)
    
#     # Assuming you want to convert it to a numpy array
#     X_numpy = X.to_numpy(dtype=np.float32)  # Specify the dtype as float32 for PyTorch compatibility
    
#     return X_numpy

# The function now returns a NumPy
def load_graph_node_features_geo(path, feature1='checkin_cnt', feature2='cat_id',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path, delimiter=',') 
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()
    # Apply geohash encoding
    geohash = df.apply(lambda row: geohash_encode(row[feature3], row[feature4]), axis=1)

    # 为了保证每次结果一致，对unique()后的geohash进行排序
    geohash_unique_sorted = sorted(geohash.unique())

    # 创建一个唯一ID映射，由于现在是排序后的结果，将保持一致性
    geohash_to_id = {geohash: idx for idx, geohash in enumerate(geohash_unique_sorted)}

    # 映射geohash到ID
    geohash_id = geohash.map(geohash_to_id)
    geohash_list = geohash_id.tolist()

    return X,geohash_list
import pandas as pd

# 读取商家数据
df = pd.read_json('/data/CaiZhuaoXiao/yelp/FL/FL_business.json', lines=True)
# 获取唯一的business_id列表
x = df['business_id'].unique()
print(len(x))

# 读取评论数据
data = pd.read_json('/data/CaiZhuaoXiao/yelp/yelp_academic_dataset_review.json', lines=True)
# 从评论数据中过滤出属于x列表中business_id的记录
filtered_data = data[data['business_id'].isin(x)]
z = filtered_data['business_id'].unique()
print(len(z))

# 计算每个user_id在这个新数据集中的记录数
user_counts = filtered_data['user_id'].value_counts()

# 保留那些至少有20条记录的user_id
sufficient_user_reviews = user_counts[user_counts >= 20].index
top_reviews_by_user = filtered_data[filtered_data['user_id'].isin(sufficient_user_reviews)]

# 再次计算每个business_id在这个新数据集中的记录数
business_counts = filtered_data['business_id'].value_counts()

# 保留那些至少有20条记录的business_id
sufficient_business_reviews = business_counts[business_counts >= 20].index
top_reviews_by_user = top_reviews_by_user[top_reviews_by_user['business_id'].isin(sufficient_business_reviews)]
# 对filtered_data进行分组排序，取每个user_id的前50条评论
top_reviews = top_reviews_by_user.groupby('user_id').apply(lambda x: x.sort_values('date', ascending=True).head(50)).reset_index(drop=True)

# 保存最终的DataFrame到JSON文件
top_reviews.to_json('/data/CaiZhuaoXiao/yelp/FL/FL_reviews.json', orient='records', lines=True)
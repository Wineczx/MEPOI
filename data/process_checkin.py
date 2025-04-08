import pandas as pd

# 读取JSON文件
df = pd.read_json('/data/CaiZhuaoXiao/yelp/FL/FL_checkin_summary.json', lines=True)

# 计算每个user_id的记录数
state_counts = df['user_id'].value_counts()

# 删除记录数小于10的user_id
df = df[~df['user_id'].isin(state_counts[state_counts < 10].index)]
t = df['user_id'].unique()
print(len(t))
df2 = pd.read_json('/data/CaiZhuaoXiao/yelp/FL/FL_business.json',lines=True)
df = pd.merge(df, df2[['business_id', 'categories','latitude','longitude']], on='business_id', how='left')
df.to_json('/data/CaiZhuaoXiao/yelp/FL/FL_checkin.json',orient='records',lines=True)
# 输出处理后的数据框

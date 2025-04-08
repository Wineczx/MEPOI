import pandas as pd
from datetime import datetime

data = pd.read_json('/data/CaiZhuaoXiao/yelp/FL/FL_checkin.json', lines=True)

data = data.sort_values(['user_id', 'date'])
data['cat_id'] = pd.factorize(data['categories'])[0] + 1
data['timestamp'] = pd.to_datetime(data['date']).apply(lambda x: datetime.timestamp(x))

# 根据 'user_id' 的总数进行筛选
# state_counts = data['user_id'].value_counts()
# df = data[~data['user_id'].isin(state_counts[state_counts < 20].index)]
# 根据 'business_id' 的总数进行筛选
# business_counts = df['business_id'].value_counts()
# data = df[df['business_id'].isin(business_counts[business_counts > 20].index)]


unique_user_ids = data['user_id'].unique()
train_data = pd.DataFrame()
test_data = pd.DataFrame()
val_data = pd.DataFrame()

for user_id in unique_user_ids:
    user_data = data[data['user_id'] == user_id]
    user_data = user_data.head(50).sort_values('date')
    print(user_id)
    sixty_percent = int(len(user_data) * 0.6)

    train_user = user_data[:sixty_percent]
    test_user_val = user_data[sixty_percent:]
    half_len = len(test_user_val) // 2

    test_user = test_user_val[:half_len]
    val_user = test_user_val[half_len:]

    train_data = pd.concat([train_data, train_user])
    test_data = pd.concat([test_data, test_user])
    val_data = pd.concat([val_data, val_user])

train_business_ids = train_data['business_id'].unique()

test_data = test_data[test_data['business_id'].isin(train_business_ids)]
val_data = val_data[val_data['business_id'].isin(train_business_ids)]

train_data = train_data.sort_values('date')
test_data = test_data.sort_values('date')
val_data = val_data.sort_values('date')

train_data.to_csv('/data/CaiZhuaoXiao/yelp/train.csv', index=False, sep='|')
test_data.to_csv('/data/CaiZhuaoXiao/yelp/test.csv', index=False, sep='|')
val_data.to_csv('/data/CaiZhuaoXiao/yelp/val.csv', index=False, sep='|')
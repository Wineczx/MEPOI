import pandas as pd

df = pd.read_json('/data/CaiZhuaoXiao/yelp/FL/FL_business.json', lines=True)
x = df['business_id'].unique()
print(len(x))

data = pd.read_json('/data/CaiZhuaoXiao/yelp/photos.json', lines=True)
df2 = pd.DataFrame(data)

filtered_data = df2[df2['business_id'].isin(x)]
z = filtered_data['business_id'].unique()
print(len(z))

filtered_data.to_json('/data/CaiZhuaoXiao/yelp/FL/FL_photos.json', orient='records', lines=True)
import pandas as pd

df = pd.read_json('/data/CaiZhuaoXiao/yelp/PAA/image_description.json', lines=True)
df2 = pd.read_json('/data/CaiZhuaoXiao/yelp/PAA/PA_business_core.json', lines=True)
df3 = pd.read_json('/data/CaiZhuaoXiao/yelp/PAA/PA_reviews_summary5.json',lines=True)
x = df3['business_id'].unique()
y = df['business_id'].unique()

filtered_data = df3[df3['business_id'].isin(y)]
filtered_data.to_json('/data/CaiZhuaoXiao/yelp/PAA/PA_checkin_summary.json', orient='records', lines=True)
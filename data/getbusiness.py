import pandas as pd
import json
df = pd.read_json('/data/CaiZhuaoXiao/yelp/yelp_academic_dataset_business.json',lines=True)
state_counts = df['state'].value_counts()
print(state_counts)
filtered_df = df[df['state'] == 'PA']
print(len(filtered_df))
filtered_df.to_json('/data/CaiZhuaoXiao/yelp/PAA/PA_business.json', orient='records', lines=True)
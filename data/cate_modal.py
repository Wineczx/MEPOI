import json
import pandas as pd

# 读取包含目标数据的JSON文件
df = pd.read_json('/data/CaiZhuaoXiao/yelp/PAA/PA_business_core.json', lines=True)

# 获取所有的unique business_id
unique_business_ids = df['business_id'].unique()

# 逐行写入JSON文件
with open('/data/CaiZhuaoXiao/yelp/PAA/PA_categories.json', 'w') as f:
    for business_id in unique_business_ids:
        categories = df[df['business_id'] == business_id]['categories'].tolist()[0]
        categories_list = [cat.strip() for cat in categories.split(',')]
        data = {business_id: categories_list}
        json.dump(data, f)
        f.write('\n')

print("写入完成！")
import json
import pandas as pd

# 读取包含目标数据的JSON文件
df = pd.read_json('/data/CaiZhuaoXiao/yelp/PAA/PA_checkin.json', lines=True)

# 获取所有的unique business_id
unique_business_ids = df['business_id'].unique()

# 逐行写入JSON文件
with open('/data/CaiZhuaoXiao/yelp/PAA/review_summary.json', 'w') as f:
    for business_id in unique_business_ids:
        texts = df[df['business_id'] == business_id]['summary'].tolist()
        data = {business_id: texts}
        json.dump(data, f)
        f.write('\n')

print("写入完成！")
import pandas as pd
import os
import shutil

df = pd.read_json('/data/CaiZhuaoXiao/yelp/FL/FL_photos.json', lines=True)
source_folder = '/data/CaiZhuaoXiao/yelp/photos'  # 设置原始图片所在的文件夹路径
target_folder = '/data/CaiZhuaoXiao/yelp/FL/photos'  # 设置目标文件夹路径
business_ids = df['business_id'].unique()
os.makedirs(target_folder, exist_ok=True)

for business_id in business_ids:
    photo_ids = df[df['business_id'] == business_id]['photo_id']
    for i, photo_id in enumerate(photo_ids, start=1):
        source_file = os.path.join(source_folder, photo_id + '.jpg')
        target_file = os.path.join(target_folder, f"{business_id}~{i}.jpg")
        shutil.copy(source_file, target_file)
        # 更新photo.json文件中的photo_id
        df.loc[(df['business_id'] == business_id) & (df['photo_id'] == photo_id), 'photo_id'] = f"{business_id}~{i}"

df.to_json('/data/CaiZhuaoXiao/yelp/FL/FL_photos.json', orient='records', lines=True)
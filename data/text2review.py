import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 载入分词器和模型
tokenizer = AutoTokenizer.from_pretrained("/data/flan-alpaca-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("/data/flan-alpaca-xl").to(device)

# 统计文本的单词数量
def word_count(text):
    return len(text.split())

# 生成摘要的函数
def generate_summary(texts):
    summaries = []
    for text in texts:
        # Check if the text has more than 20 words
        # Add the prompt
        prompt = f"Turn the following description of the content in an image into a user comment about the place in 20 words or less:{text}"
        inputs = tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=512).to(device)  # Ensure you are passing the correct arguments here
        summary_ids = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

# 尝试读取已处理的review_id，如果不存在则继续
# try:
#     processed_df = pd.read_json('/data/yelp/FL/FL_reviews_summary.json', lines=True)
#     processed_ids = set(processed_df['review_id'].tolist())
# except (ValueError, FileNotFoundError):
#     processed_ids = set()
processed_ids = set()

# 读取原始数据集
df = pd.read_json('/data/yelp/PA/image_description.json', lines=True)

batch_size = 20
num_samples = len(df['text'])

# 开始生成摘要并将它们添加到数据集
# Start generating summaries and adding them to the dataset
with open('/data/yelp/PAA/PA_photostoreview.json', 'a') as outfile:
    for i in tqdm(range(0, num_samples, batch_size), position=0, leave=True):
        # Select the batch with a copy to avoid SettingWithCopyWarning
        batch_df = df.iloc[i:i + batch_size].copy()

        # Skip batch if all its review_ids have been processed
        if set(batch_df['photo_id']).issubset(processed_ids):
            continue

        batch_texts = batch_df['text'].tolist()
        batch_summaries = generate_summary(batch_texts)

        # Add summaries back to the dataframe
        batch_df['review'] = batch_summaries  # Here we avoid using .loc for assigning to avoid the warning

        # Drop the original 'text' column
        batch_df = batch_df.drop(columns=['text'])

        # Write the processed batch to file
        batch_df.to_json(outfile, orient='records', lines=True, force_ascii=False)

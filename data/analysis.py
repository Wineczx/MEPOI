import pandas as pd

# Load the datasets
train_df = pd.read_csv('/home/zhanghuaxiang/xuyang/explainable-recommendation/dataset/PAA/test.csv',delimiter = '|')
val_df = pd.read_csv('/home/zhanghuaxiang/xuyang/explainable-recommendation/dataset/PAA/val.csv',delimiter = '|')
test_df = pd.read_csv('/home/zhanghuaxiang/xuyang/explainable-recommendation/dataset/PAA/train.csv',delimiter = '|')

# Combine the datasets
combined_df = pd.concat([train_df, val_df, test_df])

# Calculate the number of unique users, POIs, and check-ins
num_users = combined_df['user_id'].nunique()
num_pois = combined_df['business_id'].nunique()
num_checkins = combined_df['review_id'].nunique()
num_seq = len(train_df['user_id'].unique()) + len(val_df['user_id'].unique()) + len(test_df['user_id'].unique())
# Calculate the total number of interactions (i.e., rows in the combined dataset)
total_interactions = len(combined_df)

# Calculate the density of the dataset
density = total_interactions / (num_users * num_pois)

# Display the results
print(f"Number of unique users: {num_users}")
print(f"Number of unique POIs: {num_pois}")
print(f"Number of check-ins: {num_checkins}")
print(f"Number of seq: {num_seq}")
print(f"Density of the dataset: {density:.6f}")

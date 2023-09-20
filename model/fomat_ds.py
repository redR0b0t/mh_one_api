import pandas as pd


train_path="/home/u131168/mh_one_api/data/train.csv"
test_path="/home/u131168/mh_one_api/data/test.csv"
submission_path="/home/u131168/mh_one_api/data/submission.csv"

train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path)
submission_data=pd.read_csv(submission_path)

train_data.rename(columns = {'test':'TEST'}, inplace = True)

print(train_data)

# instructions=pd.Series(train_data[]

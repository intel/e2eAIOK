import os
import random
import pandas as pd

n_plus, n_time, n_bkids, n_weather, n_feels = 522, 167, 126, 35, 20

os.mkdir("data_10000")
num_files = 10
for file in range(num_files):
    records = []

    for i in range(0, 10000):
        pluids = [random.randint(0, n_plus - 1) for i in range(0, 5)]
        timeidx = random.randint(0, n_time - 1)
        bkidx = random.randint(0, n_bkids - 1)
        weatheridx = random.randint(0, n_weather - 1)
        feelsBucket = random.randint(0, n_feels - 1)
        label = random.randint(0, 1)
        records.append((pluids, timeidx, bkidx, weatheridx, feelsBucket, label))

    df = pd.DataFrame(records,
                      columns=['pluids', 'timeidx', 'bkidx', 'weatheridx', 'feelsBucket', 'label'])

    with open('data_10000/' + str(file) + '.json', 'w') as f:
        f.write(df.to_json(orient='records', lines=True))

df = pd.read_json('data/0.json', orient='columns', lines=True)
print(df.head())

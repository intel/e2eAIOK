import numpy as np
import pandas as pd
import sys

file_name = sys.argv[1]
print(file_name)
bytes_per_feature = 4
tot_fea = 40
num_batch = 748
batch_size = 256
bytes_per_batch = (bytes_per_feature * tot_fea * batch_size)
with open(file_name, "rb") as f:
     raw_data = f.read(bytes_per_batch * num_batch)
     array = np.frombuffer(raw_data, dtype=np.int32)
     array = array.reshape(-1,40)
     df = pd.DataFrame(array, columns= [f"_c{i}" for i in range(40)])
     print(df[[f"_c{i}" for i in [0, 1, 2, 3, 4, 5, 6, 7, 8] + [14, 15, 16, 17, 18, 19, 20, 21, 22]]])
     #df.to_parquet(f"{file_name}.parquet")

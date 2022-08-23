import pandas as pd
import numpy as np
import time
very_start = time.time()

path = "/mnt/sdb/xinyao/2optimize/nvidia2021/3mergeall/recsys2021-intel-opt"

if __name__ == "__main__":
    df1 = pd.read_parquet(f"{path}/data/stage2_train")
    df2 = pd.read_parquet(f"{path}/data/stage2_valid")

    pred_path = f"{path}/result/lgbm_pred_stage1.csv"
    preds = pd.read_csv(pred_path)

    index_cols = ['tweet_id', 'engaging_user_id']
    df1 = df1.merge(preds, on=index_cols, how="left")
    df2 = df2.merge(preds, on=index_cols, how="left")

    df1.to_parquet(f"{path}/data/stage2_train_pred.parquet")
    df2.to_parquet(f"{path}/data/stage2_valid_pred.parquet")

    print('This notebook took %.1f seconds'%(time.time()-very_start))
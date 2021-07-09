import pickle
import sys

def fix(df):
    for col, dtype in df.dtypes.items():
        if str(dtype) == 'category' and col.endswith('_user_id'):
            df[col] = df[col].cat.codes
        elif dtype == 'float64' or dtype == 'float16':
            df[col] = df[col].astype('float32')
    # df.take() consolidates blocks in block manager
    df.take([0, 0])

with open(sys.argv[1], 'rb') as inp:
    mlst = pickle.load(inp)

cats = None
for m in mlst:
    if cats is None:
        for col in m.df_agg.columns:
            if col.endswith('_user_id'):
                cats = m.df_agg[col].dtype.categories.values
                break
    print(f'fixing {m.name}')
    fix(m.df_agg)

result = {'users': cats, 'estimators': mlst}

with open(sys.argv[2], 'wb') as out:
    pickle.dump(result, out, pickle.HIGHEST_PROTOCOL)

from tercen.client import context as ctx
import numpy as np

tercenCtx = ctx.TercenContext()

df = (
    tercenCtx
    .select(['.y', '.ci', '.ri'])
    .groupby(['.ci','.ri'], as_index=False)
    .mean()
    .rename(columns={".y":"mean"})
    .astype({".ci": np.int32, ".ri": np.int32})
)

df = tercenCtx.add_namespace(df) 
tercenCtx.save(df)

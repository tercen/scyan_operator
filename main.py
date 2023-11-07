from tercen.client import context as context
from tercen.model.base import Pair
import numpy as np
import polars as pl
import pandas as pd
import scyan, anndata
import json

from scyan.utils import _get_subset_indices


#TODO List
# Read Tercen data directly as pandas to avoid conversion later
# Create Unit Tests
# Add/Read parameter to pass to Scyan function

# http://127.0.0.1:5400/test/w/c3ffce4e7131bfb88740387170013cd3/ds/928e597a-9ced-4ef1-a985-eb1180fe19b4
# ctx = context.TercenContext(workflowId="e2a9ef1dc04be286be60a5082d013ce8", stepId="928e597a-9ced-4ef1-a985-eb1180fe19b4")

ctx = context.TercenContext()

# if(is.null(ctx$task)) {
#   stop("task is null")
# } else {
#   pair <- Find(function(pair) identical(pair$key, "task.siblings.id"), ctx$task$environment)
#   task_siblings_id <- jsonlite::fromJSON(pair$value)
#   ctx2 <- tercenCtx(taskId = task_siblings_id)
# }
if not ctx.task is None:
    envPairs = ctx.task.environment
    ctx.log("Printing environment")
    for e in envPairs:
        if isinstance(e, Pair):
            ctx.log(str(e.key))
            if str(e.key) == "task.siblings.id":
                ctx.log("Found task ID {}".format(e.value))
                ctx.log("{}".format(e.value))
                ctx.log("{}".format(e.value[0]))
                ctx2 = context.TercenContext(taskId=e.value)
 

                ctx.log(str(ctx2))

                if ctx2 is None:
                    ctx.log("Failed to create context 2")

 
                #docker run --rm --init --memory 1124m --memory-swap 1124m --memory-swappiness 0 --name=tercen_task_a9a0da8626f5c8b833542660f200a193 --env TERCEN_SERVICE_URI=http://172.42.0.42:5400 --env TERCEN_WEB_APP_PORT=8080 tercen/scyan_operator:latest --taskId a9a0da8626f5c8b833542660f200a193 --serviceUri http://172.42.0.42:5400 --token eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL3RlcmNlbi5jb20iLCJleHAiOjE2OTkxMTE2ODcsImRhdGEiOnsiZCI6IiIsInUiOiJ0ZXN0IiwiZSI6MTY5OTExMTY4NzA4NH19.xvySAwvnrv-Sg9LY6lH0vwYA8tPBPNHLQOuuU8MQMP0[0m
else:
    
    ctx2 = context.TercenContext(workflowId="e2a9ef1dc04be286be60a5082d013ce8", stepId="5d51ba5d-8fba-4978-9fc1-3b5d2ccbe995")

ctx.log("After init")
ctx.log(ctx2.names)
ctx.log(ctx2.cnames)



# http://127.0.0.1:5400/test/w/c3ffce4e7131bfb88740387170013cd3/ds/5d51ba5d-8fba-4978-9fc1-3b5d2ccbe995
# ctx2 = context.TercenContext(workflowId="c3ffce4e7131bfb88740387170013cd3", stepId="5d51ba5d-8fba-4978-9fc1-3b5d2ccbe995")


yDf = ctx.select([".y", ".ci", ".ri"])
colDf = ctx.cselect([""])
colDf = colDf.with_columns(pl.Series(name=".ci", values=range(0,len(colDf) ), dtype=pl.Int32) )

rowDf = ctx.rselect([""])
rowDf = rowDf.with_columns(pl.Series(name=".ri", values=range(0,len(rowDf)), dtype=pl.Int32))

yDf = yDf.join(colDf, on=".ci").join(rowDf, ".ri")
yDf = yDf.drop([".ri", ".ci"])

annDf = ctx2.select([".y", ".ci", ".ri"])

annColDf = ctx2.cselect([""])
annColDf = annColDf.with_columns(pl.Series(name=".ci", values=range(0,len(annColDf)), dtype=pl.Int32))

annRowDf = ctx2.rselect([""])
annRowDf = annRowDf.with_columns(pl.Series(name=".ri", values=range(0,len(annRowDf)), dtype=pl.Int32))

annDf = annDf.join(annColDf, on=".ci").join(annRowDf, ".ri")
annDf = annDf.drop([".ri", ".ci"])



annDfP = annDf.pivot(columns=annDf.columns[1], index=annDf.columns[2], values=annDf.columns[0])
annDfP = annDfP.with_columns(pl.all().fill_null(strategy="zero"))


yDfP = yDf.pivot(columns=yDf.columns[2], index=yDf.columns[1], values=yDf.columns[0]  ) #[:,1:]


markers = np.intersect1d(yDfP.columns[1:], annDfP.columns[1:])
population = annDfP[:,0].to_numpy()
 

adata = anndata.AnnData(  yDfP.to_numpy()[:,1:].astype(np.float32) )

adata.var = pd.DataFrame(yDfP.columns[1:]).rename(columns={0:"Markers"})
adata.var_names = yDfP.columns[1:]

# FIXME HArcoded columns
adata.obs = pd.DataFrame(yDfP["asinh..rowId"]).rename(columns={0:"Observation"})
#tadata.obs_names =  tmp["Observation"].to_numpy() 


tablePd = annDfP.select(markers).to_pandas()
tablePd.index = annDfP["Population"].to_numpy()

model = scyan.Scyan(adata=adata, table=tablePd )
model.fit()


model.predict()



not_na = ~model.adata.obs["scyan_pop"].isna()
indices = _get_subset_indices(not_na.sum(), 200000)
indices = np.where(not_na)[0][indices]

latent = True
x = model(indices).numpy(force=True) if latent else model.adata[indices].X
columns = model.var_names if latent else model.adata.var_names
presentPopNames = model.adata[indices].obs["scyan_pop"].cat.categories.to_list()



#scyan.plot.pop_expressions(model, model.pop_names[4])
outDf = None

prob_name = "Prob"
for i in range(0, len(presentPopNames)):
    popName = presentPopNames[i]

    u = model(model.adata.obs["scyan_pop"] == popName)
    log_probs = model.module.prior.log_prob_per_marker(u)
    mean_log_probs = log_probs.mean(dim=0).numpy(force=True)

    df_probs = pd.DataFrame(
        mean_log_probs,
        columns=model.var_names,
        index=model.pop_names,
    )
    df_probs = df_probs.reindex(
        df_probs.mean().sort_values(ascending=False).index, axis=1
    )
    means = df_probs.mean(axis=1)
    means = means / means.min() * df_probs.values.min()
    df_probs.insert(0, prob_name, means)
    df_probs.insert(1, "InterpretPop", popName)
    df_probs.insert(2, "Population", model.pop_names)

    df_probs.sort_values(by=prob_name, inplace=True, ascending=False)

    rowNames = rowDf[rowDf.columns[0]]

    for ri in range(0, len(rowNames)):
        if not rowNames[ri] in df_probs:
            continue
        pDf = df_probs[rowNames[ri]]
        for k in range(0, len(pDf)):
            df = pd.DataFrame([[ri, pDf[k],  popName, model.pop_names[k] ]], columns=['.ri','LogProb','InterpretPop', 'Population'])
            df = df.astype({".ri": np.int32, "LogProb": np.float64})
            
            if outDf is None:
                outDf = df
            else:
                outDf = pd.concat([outDf, df])



outDf = ctx.add_namespace(outDf) 
ctx.save(outDf)

#ctx = tercenCtx()
#ctx$task$siblings$id
# MultiStep step2
# http://127.0.0.1:5400/test/w/462bec31fcad0c7eb8af65440e003fc9/ds/3edda5da-633d-42ab-bf20-2488e909b21e
# ctx2 = tercenCtx(workflowId = "462bec31fcad0c7eb8af65440e003fc9", stepId = "3edda5da-633d-42ab-bf20-2488e909b21e")



# df = (
#     tercenCtx
#     .select(['.y', '.ci', '.ri'])
#     .groupby(['.ci','.ri'], as_index=False)
#     .mean()
#     .rename(columns={".y":"mean"})
#     .astype({".ci": np.int32, ".ri": np.int32})
# )

# df = tercenCtx.add_namespace(df) 
# tercenCtx.save(df)

from tercen.client import context as context
from tercen.util import helper_functions as hlp
from tercen.model.base import Pair
import numpy as np
import polars as pl
import pandas as pd
import scyan, anndata
import matplotlib, string
matplotlib.use('Agg')

from scyan.utils import _get_subset_indices


#TODO List
# Read Tercen data directly as pandas to avoid conversion later
# Create Unit Tests
# Add/Read parameter to pass to Scyan function

# http://127.0.0.1:5400/test/w/e2a9ef1dc04be286be60a5082d013ce8/ds/928e597a-9ced-4ef1-a985-eb1180fe19b4
# 

ctx = context.TercenContext()

if not ctx.task is None:
    envPairs = ctx.task.environment

    for e in envPairs:
        if isinstance(e, Pair):
            ctx.log(str(e.key))
            if str(e.key) == "task.siblings.id":                
                nChar = len(e.value)
                taskId = e.value[2:(nChar-2)]
  
               
                ctx2 = context.TercenContext(taskId=taskId)

                if ctx2 is None:
                    ctx.log("Failed to create context 2")

 
                
else:
    # TODO Raise error
    ctx2 = None
    # ctx2 = context.TercenContext(workflowId="e2a9ef1dc04be286be60a5082d013ce8", stepId="5d51ba5d-8fba-4978-9fc1-3b5d2ccbe995")



# ctx = context.TercenContext(workflowId="e2a9ef1dc04be286be60a5082d013ce8", stepId="928e597a-9ced-4ef1-a985-eb1180fe19b4")
# # http://127.0.0.1:5400/test/w/c3ffce4e7131bfb88740387170013cd3/ds/5d51ba5d-8fba-4978-9fc1-3b5d2ccbe995
# ctx2 = context.TercenContext(workflowId="e2a9ef1dc04be286be60a5082d013ce8", stepId="5d51ba5d-8fba-4978-9fc1-3b5d2ccbe995")


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

annRowDf = annRowDf.drop([".ri"])
annColDf = annColDf.drop([".ci"])


annDfP = annDf.pivot(columns=annColDf.columns[0], index=annRowDf.columns[0], values=".y")
annDfP = annDfP.with_columns(pl.all().fill_null(strategy="zero"))


yDfP = yDf.pivot(columns=yDf.columns[2], index=yDf.columns[1], values=yDf.columns[0]  ) #[:,1:]


markers = np.intersect1d(yDfP.columns[1:], annDfP.columns[1:])
population = annDfP[:,0].to_numpy()


adata = anndata.AnnData(  yDfP.to_numpy()[:,1:].astype(np.float32) )

adata.var = pd.DataFrame(yDfP.columns[1:]).rename(columns={0:"Markers"})
adata.var_names = yDfP.columns[1:]

adata.obs = pd.DataFrame(yDfP[yDf.columns[1]]).rename(columns={0:"Observation"})



tablePd = annDfP.select(markers).to_pandas()
tablePd.index = annDfP["Population"].to_numpy()

priorSd = ctx.operator_property('PriorSD', typeFn=float, default=0.3)
lr = ctx.operator_property('LR', typeFn=float, default=0.0005)
nLayers = ctx.operator_property('Layers', typeFn=int, default=7)
nHiddenLayers = ctx.operator_property('Hidden Layers', typeFn=int, default=6)
hiddenSz = ctx.operator_property('Hidden Size', typeFn=int, default=16)
temperature = ctx.operator_property('Temperature', typeFn=float, default=0.5)
moduloTemp = ctx.operator_property('Modulo Temp', typeFn=int, default=3)
batchSize = ctx.operator_property('Batch Size', typeFn=int, default=8192)
warmUp = ctx.operator_property('WarmUp', typeFn=str, default="(0.35,4)")
w1 = float(warmUp.split(",")[0].replace("(", "").strip())
w2 = float(warmUp.split(",")[1].replace(")", "").strip())

model = scyan.Scyan(adata=adata, table=tablePd, \
                    prior_std=priorSd, lr=lr, n_layers=nLayers, \
                    n_hidden_layers=nHiddenLayers, \
                    hidden_size=hiddenSz, temperature=temperature, \
                    batch_size=batchSize, modulo_temp=moduloTemp, \
                    warm_up=(w1, w2) )

ctx.log("Performing model fit")
model.fit()

ctx.log("Performing predict")
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

ctx.log("Creating outDf")
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
            df = pd.DataFrame([[ri, pDf.iloc[k],  popName, model.pop_names[k] ]], columns=['.ri','LogProb','InterpretPop', 'Population'])
            df = df.astype({".ri": np.int32, "LogProb": np.float64})
            
            if outDf is None:
                outDf = df
            else:
                outDf = pd.concat([outDf, df])

#TODO Turn into relation
ctx.log("Saving outDf")
outDf = ctx.add_namespace(outDf) 

ctx.save(outDf)




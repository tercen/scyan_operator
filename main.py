from tercen.client import context as context
from tercen.util import helper_functions as hlp
from tercen.model.base import Pair
import numpy as np
import polars as pl
import pandas as pd
import scyan, anndata
import matplotlib
import math
matplotlib.use('Agg')

from scyan.utils import _get_subset_indices

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
    ctx2 = None

yDf = ctx.select([".y", ".ci", ".ri"])
colDf = ctx.cselect([""])
colDf = colDf.select('obs_' + pl.col(colDf.columns[0]).cast(pl.Utf8))
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
adata.obs_names = yDfP[yDf.columns[1]]


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

ctx.log("Fitting model...")
model.fit()

ctx.log("Predicting cell populations...")
model.predict()

fakeSeries = model.adata.obs["scyan_pop"] != 'NoPop'
fakeSeries[0] = True
u = model(fakeSeries)
logProbs = model.module.prior.log_prob(u)


ctx.log("Creating output table")
outDf = None
dfList = [None] * len(adata.obs_names)
dfList2 = [None] * (len(adata.obs_names)*len(population))
idx = 0
for i in range(0, len(adata.obs_names)):
    
    model.adata.obs["scyan_pop"].iloc[i].__class__
    if isinstance( model.adata.obs["scyan_pop"].iloc[i], str ):
        pop = model.adata.obs["scyan_pop"].iloc[i] 
    else:
        pop = "None"
    tmpDf = pl.DataFrame({".ci":int(i), "PredictedPopulation":pop, \
                          "MaxLogProb":np.max(logProbs[i,:].tolist())  })
    logPops = logProbs[i,:].tolist()
    probs = (np.exp(logPops) / (np.exp(logPops)).sum()).tolist()
    for p in population:
        tmpDf2 = pl.DataFrame({".ci":int(i), "Population":p, \
                               "LogProb":logPops.pop(0), \
                                "Prob":probs.pop(0)})

        dfList2[idx] = tmpDf2
        idx = idx + 1
        
    dfList[i] = tmpDf

outDf =  pl.concat(dfList)
outDf2 =  pl.concat(dfList2)
dfList.clear()
dfList2.clear()

outDf = outDf.with_columns(pl.col(".ci").cast(pl.Int32))

outDf2 = outDf2.with_columns(pl.col(".ci").cast(pl.Int32))

ctx.log("Saving outDf")
outDf = ctx.add_namespace(outDf) 
outDf2 = ctx.add_namespace(outDf2) 

ctx.save([outDf, outDf2])

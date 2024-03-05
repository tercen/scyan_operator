from tercen.client import context as context
from tercen.model.impl import Pair
import numpy as np
import polars as pl
import pandas as pd
import scyan, anndata
import matplotlib
matplotlib.use('Agg')

import re

def tercen_int(x):
    ptr = re.compile('[0-9\.]+', re.UNICODE)
    if isinstance(x, int):
        return x
    
    if isinstance(x, str):
        return int("".join(["" if ptr.match(c) is None else c for c in x]).split(".")[0])
    else:
        return int(x)
    
def tercen_float(x):
    ptr = re.compile('[0-9\.]+', re.UNICODE)
    if isinstance(x, int):
        return x
    
    if isinstance(x, str):
        return float("".join(["" if ptr.match(c) is None else c for c in x])[0])
    else:
        return float(x)

def tercen_bool(x):
    return x == 'true'





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
                    exit(code=0) 

else:
    ctx.log("Failed to create context 2")
    exit(code=0) 
    ctx2 = None




fullOutput = ctx.operator_property('FullOutput', typeFn=tercen_bool, default=False)
priorSd = ctx.operator_property('PriorSD', typeFn=tercen_float, default=0.3)
lr = ctx.operator_property('LR', typeFn=tercen_float, default=0.0005)
nLayers = ctx.operator_property('Layers', typeFn=tercen_int, default=7)
nHiddenLayers = ctx.operator_property('Hidden Layers', typeFn=tercen_int, default=6)
hiddenSz = ctx.operator_property('Hidden Size', typeFn=tercen_int, default=16)
temperature = ctx.operator_property('Temperature', typeFn=tercen_float, default=0.5)
moduloTemp = ctx.operator_property('Modulo Temp', typeFn=tercen_int, default=3)
batchSize = ctx.operator_property('Batch Size', typeFn=tercen_int, default=8192)
warmUp = ctx.operator_property('WarmUp', typeFn=str, default="(0.35,4)")
w1 = float(warmUp.split(",")[0].replace("(", "").strip())
w2 = float(warmUp.split(",")[1].replace(")", "").strip())


yDf = ctx.select([".y", ".ci", ".ri"])
colDf = ctx.cselect([""])


colDf = colDf.select('o' + pl.col(colDf.columns[0]).cast(pl.Int32).cast(pl.Utf8))
colDf = colDf.with_columns(pl.Series(name=".ci", values=range(0,len(colDf) ), dtype=pl.Int32) )


rowDf = ctx.rselect([""])
rowDf = rowDf.with_columns(pl.Series(name=".ri", values=range(0,len(rowDf)), dtype=pl.Int32))

yDf = yDf.join(colDf, on=".ci")\
            .drop(".ci")\
            .join(rowDf, ".ri")\
            .drop(".ri")


annDf = ctx2.select([".y", ".ci", ".ri"])

annColDf = ctx2.cselect([""])
annColDf = annColDf.with_columns(pl.Series(name=".ci", values=range(0,len(annColDf)), dtype=pl.Int32))

annRowDf = ctx2.rselect([""])
annRowDf = annRowDf.with_columns(pl.Series(name=".ri", values=range(0,len(annRowDf)), dtype=pl.Int32))

annDf = annDf.join(annColDf, on=".ci").join(annRowDf, ".ri")
annDf = annDf.drop([".ri", ".ci"])

annRowDf = annRowDf.drop([".ri"])
annColDf = annColDf.drop([".ci"])


annDfP = annDf.pivot(columns=annColDf.columns[0], index=annRowDf.columns, values=".y", aggregate_function='first')

yDfP = yDf.pivot(columns=yDf.columns[2], index=yDf.columns[1], values=yDf.columns[0], aggregate_function='first'  ) #[:,1:]


markers = np.intersect1d(yDfP.columns[1:], annDfP.columns[len(ctx2.rnames):])
priorPop = annDfP[:,0].to_numpy()

adata = anndata.AnnData(  yDfP.to_numpy()[:,1:].astype(np.float32) )

adata.var = pd.DataFrame(yDfP.columns[1:]).rename(columns={0:"Markers"})
adata.var_names = yDfP.columns[1:]

adata.obs = pd.DataFrame(yDfP[yDf.columns[1]]).rename(columns={0:"Observation"})
adata.obs_names = yDfP[yDf.columns[1]]


# Population -> Highest variance : Leaves in the hierarchy tree
tablePd = annDfP.select(markers).to_pandas()

for i in range(0, len(ctx2.rnames)):
    if i == 0: # Always call this level population
        tablePd["Population"] = annDfP[ctx2.rnames[i]].to_numpy()
    else:
        tablePd[ctx2.rnames[i]] = annDfP[ctx2.rnames[i]].to_numpy() 


tablePd[ctx2.rnames] = tablePd[ctx2.rnames].astype('category')
tablePd = tablePd.set_index(ctx2.rnames)


model = scyan.Scyan(adata=adata, table=tablePd, \
                    prior_std=priorSd, lr=lr, n_layers=nLayers, \
                    n_hidden_layers=nHiddenLayers, \
                    hidden_size=hiddenSz, temperature=temperature, \
                    batch_size=batchSize, modulo_temp=moduloTemp, \
                    warm_up=(w1, w2) )


ctx.log("Fitting model...")
model.fit(  )


ctx.log("Predicting cell populations...")
model.predict()

ctx.log("Calculating population probabilities")
outDf = None
outDf2 = None

populations = [""]
[populations.append(p) for p in model.level_names]

for j in range(0, len(populations)):
    popLabel = populations[j]
    tercenColName = ctx2.rnames[j]
    if popLabel == "":
        outColName = "scyan_pop"
    else:
        outColName = "scyan_pop_{}".format(popLabel)

    fakeSeries = model.adata.obs[outColName] != 'NoPop'
    fakeSeries[0] = True
    logProbs = model.module.prior.log_prob(model(fakeSeries))

    dfList = [] 
    dfList2 = [] 

    
        

    predColName = "PredictedPopulation_{}".format(tercenColName)
    predMaxLogProbName  = "MaxLogProb_{}".format(tercenColName)

    for i in range(0, len(model.adata.obs_names)):
        tmpDf = pl.DataFrame({".ci":int(i) })

        if isinstance( model.adata.obs[outColName].iloc[i], str ):
            pop = model.adata.obs[outColName].iloc[i] 
        else:
            pop = "None"

        tmpDf = tmpDf.with_columns(pl.lit(pop).alias(predColName ) ).\
                    with_columns(pl.lit(np.max(logProbs[i,:].tolist())).alias(predMaxLogProbName ) )
        
        dfList.append(tmpDf)

        if fullOutput == True:
            logPops = logProbs[i,:].tolist()
            probs = (np.exp(logPops) / (np.exp(logPops)).sum()).tolist()
            tmpDf2 = pl.DataFrame({".ci":int(i) })

            predProbName = "Prob_{}".format(tercenColName)
            predLogProbName  = "LogProb_{}".format(tercenColName)

            for p in priorPop:
                tmpDf2 = tmpDf2.with_columns(pl.lit(logPops.pop(0)).alias(predLogProbName ) ).\
                            with_columns(pl.lit(probs.pop(0)).alias(predProbName ) )
                dfList2.append(tmpDf2)


    if outDf is None:
        outDf = pl.concat(dfList)
    else:
        outDf = outDf.join(  pl.concat(dfList), on=".ci" ) 
    dfList = None

    if fullOutput == True:
        if outDf2 is None:
            outDf2 = pl.concat(dfList2)
        else:
            outDf2 = outDf2.join(  pl.concat(dfList2), on=".ci" ) 
        dfList2 = None
    

    
ctx.log("Saving outDf")

outDf = outDf.with_columns(pl.col(".ci").cast(pl.Int32))
outDf = ctx.add_namespace(outDf) 

if fullOutput:
    outDf2 =  pl.concat(dfList2)
    dfList2.clear()
    outDf2 = outDf2.with_columns(pl.col(".ci").cast(pl.Int32))
    outDf2 = ctx.add_namespace(outDf2) 
    ctx.save([outDf, outDf2])

else:
    ctx.save([outDf])



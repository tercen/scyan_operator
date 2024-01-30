from tercen.client import context as context
from tercen.model.impl import Pair
import numpy as np
import polars as pl
import pandas as pd
import scyan, anndata, os
import matplotlib
matplotlib.use('Agg')

#http://127.0.0.1:5400/test/w/5409bc1875748e715c48848fd3004e42/ds/1ba15732-ba1e-400c-828d-8e542decfc5c
#http://127.0.0.1:5400/test/w/5409bc1875748e715c48848fd3004e42/ds/9deae2f2-b062-46bd-a36b-8d5c445e3d4b
# ctx = context.TercenContext(workflowId="5409bc1875748e715c48848fd3004e42",\
                            #  stepId="1ba15732-ba1e-400c-828d-8e542decfc5c")

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
    # ctx2 = context.TercenContext(workflowId="5409bc1875748e715c48848fd3004e42",\
                                #   stepId="9deae2f2-b062-46bd-a36b-8d5c445e3d4b")



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


annDfP = annDf.pivot(columns=annColDf.columns[0], index=annRowDf.columns, values=".y", aggregate_function='first')

yDfP = yDf.pivot(columns=yDf.columns[2], index=yDf.columns[1], values=yDf.columns[0], aggregate_function='first'  ) #[:,1:]


markers = np.intersect1d(yDfP.columns[1:], annDfP.columns[len(ctx2.rnames):])
population = annDfP[:,0].to_numpy()


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

tablePd[ctx2.rnames] = tablePd[ctx2.rnames].astype('category')#.cat.codes.unstack()
tablePd = tablePd.set_index(ctx2.rnames)
def tercenBool(x):
    return x == 'true'
fullOutput = ctx.operator_property('FullOutput', typeFn=tercenBool, default=False)

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
model.fit( )


ctx.log("Predicting cell populations...")
model.predict()


outDf = None
dfList = [None] * len(adata.obs_names) 
if fullOutput == True:
    dfList2 = [None] * (len(adata.obs_names)*len(population)) # 2 -> levels


popLabels = [""]
[popLabels.append(l) for l in model.level_names]


logProbsList = []
outColNames = []
tercenColNames = []
ctx.log("Calculating population probabilities")
for i in range(0, len(popLabels)):
    popLabel = popLabels[i]
    tercenColName = ctx2.rnames[i]
    if popLabel == "":
        outColName = "scyan_pop"
    else:
        outColName = "scyan_pop_{}".format(popLabel)

    outColNames.append(outColName)
    tercenColNames.append(tercenColName)
    fakeSeries = model.adata.obs[outColName] != 'NoPop'
    fakeSeries[0] = True
    u = model(fakeSeries)

    logProbsList.append( model.module.prior.log_prob(u))


ctx.log("Creating output tables")

idx = -1
    
for i in range(0, len(adata.obs_names)):
    tmpDf = pl.DataFrame({".ci":int(i) })

    for j in range(0, len(popLabels)):
        outColName = outColNames[j]
        tercenColName = tercenColNames[j]
        logProbs = logProbsList[j]
        if isinstance( model.adata.obs[outColName].iloc[i], str ):
            pop = model.adata.obs[outColName].iloc[i] 
        else:
            pop = "None"

        tmpDf = tmpDf.with_columns(pl.lit(pop).alias("PredictedPopulation_{}".format(tercenColName) ) ).\
                      with_columns(pl.lit(np.max(logProbs[i,:].tolist())).alias("MaxLogProb_{}".format(tercenColName) ) )
    
        if fullOutput:
            tmpDf2 = pl.DataFrame({".ci":int(i)})
            logPops = logProbs[i,:].tolist()
            probs = (np.exp(logPops) / (np.exp(logPops)).sum()).tolist()
            for p in population:
                                # tmpDf2 = pl.DataFrame({".ci":int(i), "Population":p, \
                                #     "LogProb":logPops.pop(0), \
                                #         "Prob":probs.pop(0),\
                                #         "Level":tercenColName})
                tmpDf2 = pl.DataFrame({".ci":int(i)})
                tmpDf2.with_columns(pl.lit(logPops.pop(0)).alias("LogProb_{}".format(tercenColName))).\
                    with_columns(pl.lit(probs.pop(0)).alias("Prob_{}".format(tercenColName))).\
                    with_columns(pl.lit(p).alias("Population_{}".format(tercenColName)))


                if j == 0:
                    idx = idx + 1
                dfList2[idx] = tmpDf2
                
        
    dfList[i] = tmpDf
    


ctx.log("Saving outDf")

outDf =  pl.concat(dfList)
dfList.clear()
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
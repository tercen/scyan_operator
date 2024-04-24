from tercen.client import context as context
from tercen.model.impl import Pair
from tercen.util.helper_functions import as_relation, as_join_operator, left_join_relation, dataframe_to_table

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



#Dbg
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
model.fit()

ctx.log("Predicting cell populations...")
model.predict()

ctx.log("Calculating population probabilities")
outDf = None
outDf2 = None

## Get predictions (Table 1)
populations = [""]
[populations.append(p) for p in model.level_names]
    
model.adata.obs[".ci"] = np.arange(len(model.adata.obs), dtype='int32')
df_pred = model.adata.obs
df_pred = (df_pred
    .rename(columns={"scyan_log_probs": "Log_Prob", "scyan_pop": "Predicted_Population"})
    .astype({'Observation': 'string', 'Predicted_Population': 'string', 'Log_Prob': 'float64'})
    .drop(['Observation'], axis=1)
    .fillna('None')
)

## Get latent expression values (Table 2) 
x = model().numpy(force=True)
columns = model.var_names
df_latent = pd.DataFrame(x, columns=columns, index=model.adata.obs.index)
df_latent["Population"] = model.adata.obs['scyan_pop']
df_latent_summ = df_latent.groupby("Population").mean().reset_index()

df_latent_out = (
    pd
    .melt(df_latent_summ, id_vars='Population', var_name='Latent_Marker', value_name='Latent_Expression')
    .rename(columns={"Population": "Latent_Population"})
    .astype({'Latent_Population': 'string', 'Latent_Expression': 'float64', 'Latent_Marker': 'string'})
)
## Get population names (Table 3) 
df_pops = ctx2.rselect(df_lib="pandas").rename(columns={"Population": "Predicted_Population"})
new_row = pd.Series([str('None')]*len(df_pops.columns), index=df_pops.columns)
df_pops.loc[-1] = new_row
df_pops.index = df_pops.index + 1
df_pops = df_pops.sort_index().astype('string')
df_pops[".id2"] = np.arange(len(df_pops), dtype='int32')

# join to higher levels
# to do

# Replace population string by index in first table
# if relation mapping
# df_pred['.id1'] = df_pred['Predicted_Population'].map(df_pops.set_index('Predicted_Population')['.id2']).astype('Int32')
# df_pred_out = df_pred.drop(['Predicted_Population'], axis=1).astype({'.id1': 'int32'})

df_pred_out = pd.merge(
    df_pred, df_pops, left_on="Predicted_Population", right_on="Predicted_Population", how="left", sort=False
)


# Format and save output
ctx.log("Saving output")

df_pred_out = ctx.add_namespace(df_pred_out) 
df_latent_out = ctx.add_namespace(df_latent_out) 
df_pops_out = ctx.add_namespace(df_pops) 

ctx.save([df_pred_out, df_latent_out, df_pops_out])

# Save as relation
rel_pred = as_relation(df_pred_out)
rel_latent = as_relation(df_latent_out)
rel_pops = as_relation(df_pops_out)

rel_preds = left_join_relation(rel_pred, rel_pops, '.id1', '.id2')

## Expected:
ctx.save_relation([as_join_operator(rel_preds, [], []), as_join_operator(rel_latent, [], [])])



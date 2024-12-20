#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel, pearsonr, spearmanr, zscore, norm
from statsmodels.stats.multitest import multipletests
from sklearn.feature_extraction.text import CountVectorizer
import statsmodels.api as sm
import statsmodels.formula.api as smf

from IPython import embed

sys.path.append("/home/msap/resources")
import idp


def ttestSummaries(df,condition_col,measure_cols,paired=None):
  """Function that compares a set of features in two groups.
  df: data containing measures and conditions
  condition_col: column name containing the group belonging (e.g., control vs. treatment)
  measure_cols: column names to compare accross groups (e.g., num words, num pronouns, etc)
  paired: None if indep. t-test, else: name of column to pair measures on.
  """
  d = {}
  
  if paired:
    df = df.loc[~df[paired].isnull()]
    df = df.sort_values(by=[paired,condition_col])
    
  for m in measure_cols:
    d[m] = ttestSummary(df,condition_col,m,paired=paired)
    
  statDf = pd.DataFrame(d).T
  statDf["p_holm"] = multipletests(statDf["p"],method="h")[1]
  return statDf

def ttestSummary(df,condition_col,measure_col,paired=None):
  # conds = sorted(list(df[condition_col].unique()))
  conds = sorted(filter(lambda x: not pd.isnull(x),df[condition_col].unique()))

  conds = conds[:2]
  assert len(conds) == 2, "Not supported for more than 2 conditions "+str(conds)
  
  a = conds[0]
  b = conds[1]
  
  ix = ~df[measure_col].isnull()
  if paired:
    # merge and remove items that don't have two pairs
    pair_counts = df[ix].groupby(by=paired)[measure_col].count()
    pair_ids = pair_counts[pair_counts == 2].index
    ix = df[paired].isin(pair_ids)
    
  s_a = df.loc[(df[condition_col] == a) & ix,measure_col]
  s_b = df.loc[(df[condition_col] == b) & ix,measure_col]

  out = {
    f"mean_{a}": s_a.mean(),
    f"mean_{b}": s_b.mean(),
    f"std_{a}": s_a.std(),
    f"std_{b}": s_b.std(),
    f"n_{a}": len(s_a),
    f"n_{b}": len(s_b),    
  }
  if paired:    
    t, p = ttest_rel(s_a,s_b)
  else:
    t, p = ttest_ind(s_a,s_b)
    
  out["t"] = t
  out["p"] = p

  # Cohen's d  
  out["d"] = (s_a.mean() - s_b.mean()) / (np.sqrt(( s_a.std() ** 2 + s_b.std() ** 2) / 2))
  
  return out

def correlSummaries(df,left,rights,method="pearsonr"):
  d = {}
  for c in rights:
    d[c] = correlSummary(df,left,c,method=method)

  statDf = pd.DataFrame(d).T
  statDf["p_holm"] = multipletests(statDf["p"],method="h")[1]
  
  return statDf

def correlSummary(df,left,right,method="pearsonr"):
  corrF = spearmanr if method == "spearmanr" else pearsonr
  ix = ~(df[left].isnull() | df[right].isnull() )
  d = df[ix]
  r,p = corrF(d[left],d[right])
  n = len(d)
  return {"n":n,"r":r,"p":p}

def controlledCorrelSummariesRight(df,lefts,right,ctrls):
  d = {}
  for c in lefts:
    d[c] = controlledCorrelSummary(df,c,right,ctrls)
    
  statDf = pd.DataFrame(d).T
  statDf["p_holm"] = multipletests(statDf["p"],method="h")[1]
  
  return statDf
 
def controlledCorrelSummaries(df,left,rights,ctrls,random_effects=[]):
  d = {}
  for c in rights:
    d[c] = controlledCorrelSummary(df,left,c,ctrls,random_effects_=random_effects)

  statDf = pd.DataFrame(d).T
  statDf["p_holm"] = multipletests(statDf["p"],method="h")[1]
  
  return statDf

def controlledCorrelSummary(df_,left_,right_,ctrls_,random_effects_=[],normalize=True):
  df = df_[[left_,right_]+ctrls_+random_effects_].rename(
    columns={c: c.replace("-","_") for c in [left_,right_]+ctrls_+random_effects_})

  left = left_.replace("-","_")
  right = right_.replace("-","_")
  ctrls = [c.replace("-","_") for c in ctrls_]
  random_effects = [c.replace("-","_") for c in random_effects_]
  
  coding = None
  if df[left].nunique() == 2:
    fam = sm.families.Binomial()
    if df[left].dtype == object:
      coding = list(enumerate(df.loc[~df[left].isnull(),left].unique()))
  else:
    fam = sm.families.Gaussian()
  
  if normalize:
    for c in df.select_dtypes([int,float]):
      if c == left: continue
      df.loc[~df[c].isnull(),c] = zscore(df.loc[~df[c].isnull(),c])
      
  form = f"{left} ~ {right}"
  if ctrls:
    form += "+" + "+".join(ctrls)
    
  try:
    if random_effects:
      print("Random effects")
      fit = sm.BinomialBayesMixedGLM.from_formula(form,{c:f"0 + C({c})" for c in random_effects},data=df).fit_vb()
      # fDf = pd.DataFrame(data=np.stack([fit.fe_mean,fit.fe_sd]).T,index=fit.model.exog_names,columns=["params","bse"])
      print(fit.summary())
      fit.params = pd.Series(fit.fe_mean,index=fit.model.exog_names)
      fit.bse = pd.Series(fit.fe_sd,index=fit.model.exog_names)
      fit.z = fit.params / fit.bse
      fit.pvalues = 1 - np.abs(fit.z).apply(norm.cdf)
      fit.pvalues2 = 2*fit.pvalues
      
      embed();exit()
    else:
      fit = smf.glm(form,data=df,family=fam,missing="drop").fit()
  except Exception as e:
    print(e)
    return {}
    # embed()
    # exit()
  
  out = {"n": fit.nobs}
  
  for ctrl, beta, p in pd.concat([fit.params,fit.pvalues],axis=1).itertuples():
    if ctrl == "Intercept": continue
    if ctrl == right:
      out.update({
        f"p": p,
        f"beta": beta
      })
    else:
      out.update({
        f"p_{ctrl}": p,
        f"beta_{ctrl}": beta
      })
  return out
  

def printStats(stats,p_thresh=0.1,sort="d",round=4,dont_show_ctrls=False):
  cols = [c for c in stats.columns if "[" not in c] if dont_show_ctrls else stats.columns
  if len(stats.query(f"p<={p_thresh}")) < 1:
    print("No significance!")
    # if len(stats) < 5:
    #   print(stats.sort_values(by=sort).round(round))
  else:
    print(stats[cols].query(f"p<={p_thresh}").sort_values(by=sort).round(round))
  print()

def dla(df,text_col="story",cond_col="memType",min_word_count=5):
  cv = CountVectorizer(min_df=min_word_count)
  
  X = cv.fit_transform(df[text_col])
  X = pd.DataFrame(data=X.todense(),index=df.index,columns=list(cv.vocabulary_))
  X[cond_col] = df[cond_col]
  
  counts = X.groupby(by=cond_col).sum().T
  
  cond0 = sorted(df[cond_col].unique())[0]
  
  deltas = idp.idp(counts,cond0)
  
  deltas.name = cond0
  return deltas

if __name__ == "__main__":
  df = pd.read_csv("processedFiles/pilotV3-13_merged.csv")
  d = dla(df[df["memType"].isin(["recalled","imagined"])],"mostSurprising")
  embed()

# IPython log file
import pandas as pd
import sys
# sys.path.append("/home/msap/tools/")

from demo_detect import predict, twokenize
from langid.langid import LanguageIdentifier, model

dialCats = ["aav","hispanic","other","white"]
lpy_identifier = None
def load_lpy_identifier():
  """Idempotent"""
  global lpy_identifier
  if lpy_identifier is not None:
    return
  lpy_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

def countDialectWords(tweet):
  if tweet is None or pd.isnull(tweet):
    props = pd.Series(index=dialCats)
  else:
    toks = twokenize.tokenizeRawTweetText(tweet)
    props = pd.Series(predict.predict(toks),index=dialCats)
    
  return props.fillna(0.0)

def detectDialect(text):
  s = countDialectWords(text)
  am = s[dialCats].idxmax()
  
  if s[am] >= .80:
    s["dialect_80pct"] = am
  else:
    s["dialect_80pct"] = "other"
    
  s["dialect_argmax"] = am

  return s

predict.load_model()
load_lpy_identifier()

if __name__ == "__main__":
  s = detectDialect("I ain't yo mama, fuck outta here")
  exit()
  df[dialCats] = df[text_col].progress_apply(countDialectWords)
  
  df["dialect_80pct"] = "other"
  df.loc[(df[dialCats]>=.8).any(axis=1),"dialect_80pct"] = \
  df.loc[(df[dialCats]>=.8).any(axis=1),dialCats].idxmax(axis=1)
  df["dialect_argmax"] = df[dialCats].idxmax(axis=1)

  print(pd.value_counts(df.dialect_80pct))
  print(pd.value_counts(df.dialect_argmax))

  outf = inf.replace(".csv","_dial.csv")
  df.to_csv(outf,index=False)
  print(f"Exported to {outf}")
  

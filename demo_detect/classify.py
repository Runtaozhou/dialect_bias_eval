from __future__ import division, print_function
import langid
from langid.langid import LanguageIdentifier, model
from sotopia.demo_detect import twokenize
from sotopia.demo_detect import predict

from IPython import embed

try:
    unicode("is this python3")
except NameError:
    unicode = str
    
def predict_lang(toks, alpha=.125, numpasses=5, thresh1=1, thresh2=0.1, thresh3=0.1):
    final_lang = ''
    posterior = predict.predict(toks, alpha, numpasses, thresh1, thresh2)
    if posterior is not None:
        if posterior[2] < thresh3:
            final_lang = 'en'
    return final_lang

lpy_identifier = None

def load_lpy_identifier():
    """Idempotent"""
    global lpy_identifier
    if lpy_identifier is not None:
        return
    lpy_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

def classify(tweet):
    # load demographic model
    predict.load_model()

    # instantiate LanguageIdentifier to normalize langid.py's output score into a probability
    load_lpy_identifier()
        
    # require that the tweet be in Unicode
    if not isinstance(tweet, unicode):
        # shouldn't happen in Python3
        tweet = tweet.decode('utf-8')

    langid_pred = lpy_identifier.classify(tweet)
    toks = twokenize.tokenizeRawTweetText(tweet)
    dem_pred = predict_lang(toks)

    if langid_pred[0] == "en" or dem_pred == "en":
        final_pred = "en"
    else:
        final_pred = "nonen"

    pred = {}
    pred['final_prediction'] = final_pred
    pred['langidpy_prediction'] = langid_pred
    return pred

if __name__=='__main__':
    pred = classify('Hello! This is a test.')
    print(pred)
    pred = classify('you are not interested in this do you?')
    print(pred)

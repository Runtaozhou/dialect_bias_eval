'''
Util functions
'''

import re
import os
import time
import math
from tqdm import tqdm
import pandas as pd
import evaluate 
import random
import datasets
import convokit
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Set
from random import sample 
from collections import defaultdict
from collections import Counter
from readability.readability import Readability
from multivalue import Dialects
from pydantic import BaseModel, validator
from llm_chain import create_chain
from convokit import Corpus, Speaker, Utterance
from convokit import download
from convokit import Classifier
from convokit import PolitenessStrategies
from convokit import TextParser
from scipy.sparse import csr_matrix
import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# Define the pydantic model
class DialectDetectionOutput(BaseModel):
    dialect: str

    @validator('dialect')
    def check_dialect(cls, value):
        if value not in ['aav', 'white']:
            raise ValueError("Output must be either 'aav' for AAVE or 'white' for SAE.")
        return value

'''
another way to classify the sentences into AAVE or SAE  using LLM
'''

def dialect_detection_llm(model_name, sentence):
    # Define the prompt template for multiple-choice questions
    template = (
        "Please look through the sentence from a dialogue carefully: {sentence} and determine if the sentence "
        "belongs to AAVE or SAE. If you think the sentence is AAVE, output 'The dialect is: aav', "
        "if you think the sentence is SAE, output 'The dialect is: white'."
    )
    input_variables = ["sentence"]

    # Create chain (assuming create_chain is a pre-defined function in your code)
    chain = create_chain(model_name, template, input_variables)
    input_data = {"sentence": sentence}
    
    # Invoke the questions
    response = chain.invoke(input_data)
    text = response['text']
    
    # Use regex to match the output
    pattern = r"The dialect is:\s*(\w+)"
    match = re.search(pattern, text)
    
    if match:
        dialect_result = match.group(1).lower()
    else:
        dialect_result = "Unable to find the match"

    # Use pydantic model to validate the output
    valid_output_flag = False
    retry = 0
    while(valid_output_flag == False and retry <=4):
        try:
            validated_output = DialectDetectionOutput(dialect=dialect_result)
            valid_output_flag = True
        except ValueError as e:
            retry +=1
            response = chain.invoke(input_data)
            text = response['text']
            # Use regex to match the output
            pattern = r"The dialect is:\s*(\w+)"
            match = re.search(pattern, text)
            if match:
                dialect_result = match.group(1).lower()
            else:
                dialect_result = "Unable to find the match"
    if valid_output_flag == True:
        final_output = validated_output.dialect
    else:
        final_output = 'other'
    return final_output
# Define the pydantic model for SAE translation output
class SAETranslationOutput(BaseModel):
    translated_sentence: str

'''
this function translate the aave question back to sae question, giving the model name and the question
'''
def aave_to_sae_translation(model_name, question):
    # Define the prompt template for translation
    template = (
        "Please translate the following multiple choice question from African American Venacular English(AAVE) question to Standard American English(SAE) question: '{question}'. "
        "Be sure to translate and include the options of the translated multiple choice question in the final output"
        "Output the result in this exact format: 'The translated question is: [SAE question]'."
    )
    input_variables = ["question"]
    
    # Create chain (assuming create_chain is a pre-defined function in your code)
    chain = create_chain(model_name, template, input_variables)
    input_data = {"question": question}
    
    # Invoke the translation request
    response = chain.invoke(input_data)
    text = response['text']
    identifier = "The translated question is: "
    if "The translated question is" in text:
        translated_sentence = text.replace(identifier, "")
    else:
        translated_sentence = text
    
    return translated_sentence

'''
calculate the perplexity of model on a sentence based on model id (in huggying face)
'''
def perplexity(model_id, sentence):
    perplexity = evaluate.load("perplexity", module_type="metric")
    input_texts = sentence
    results = perplexity.compute(model_id=model_id,
                                 add_start_token=False,
                                 predictions=input_texts)
    return results

'''
calculate the answer accuracy of a model in MMLU questions. 
'''
def extract_model_accuracy(df):
    df['letter_answer'] = df['letter_answer'].str.replace(r'[()]', '', regex=True)
    matches_df = (df['letter_answer'] == df['correct_answer']).sum()/len(df)
    return matches_df 



'''
###################################
build the politeness classifier with the corpus of widipedia and stack exchange 
params: 
N/A
return:
a politeness classifer
###################################
'''   
def build_politeness_classifier():
    print("initializing the classifier...")
    random.seed(10)
    parser = TextParser(verbosity=1000)
    ps = PolitenessStrategies()
    ## train the classifier on the corpus of widipedia and stack exchange
    # load the corpus
    wiki_corpus = Corpus(filename=download("wikipedia-politeness-corpus"))
    utterance_list = [wiki_corpus.get_utterance(i) for i in wiki_corpus.get_utterance_ids()]
    stack_corpus = Corpus(filename=download("stack-exchange-politeness-corpus"))
    combine_corpus = stack_corpus.add_utterances(utterance_list)
    neutral_utterances = [utt for utt in combine_corpus.iter_utterances() if utt.meta["Binary"] == 0]
    polite_utterances = [utt for utt in combine_corpus.iter_utterances() if utt.meta["Binary"] == 1]
    # balance the training dataset. 
    sampled_neutral1 = sample(neutral_utterances,len(polite_utterances))
    sampled_neutral1.extend(polite_utterances)
    combine_corpus_polite = Corpus(utterances=sampled_neutral1)
    combine_corpus_polite = parser.transform(combine_corpus_polite)
    combine_corpus_polite = ps.transform(combine_corpus_polite, markers=True)
    labeller_polite = lambda utt: 0 if utt.meta['Binary'] == 0 else 1
    clf_polite = Classifier(obj_type="utterance", 
                            pred_feats=["politeness_strategies"], 
                            labeller=labeller_polite,
                            )
    clf_polite.fit(combine_corpus_polite)
    return clf_polite
        
'''
###################################
classify the sentences as the polite or impolite based on text in the standard qna df in qna dataset

params: 
df - pd.DataFrame object that stores the questions and the answers
text_attribute: -str, the column name in that df that stores all the answers. default to "answer"

return:
a list that have the number of the sentences predicted as polite and the number of sentences predicted as impolite. [num_polite, num_impolite]
###################################
'''        
def predict_politeness(clf, df, text_attribute : str = "answer"):
    speaker = Speaker()
    parser = TextParser(verbosity=1000)
    ps = PolitenessStrategies()
    utter_lst = []
    for i in range(len(df)):
        utter = Utterance(speaker =speaker, text = df.loc[i][text_attribute], id = str(i))
        utter_lst.append(utter)
    corpus = Corpus(utterances =utter_lst)
    test_ids = corpus.get_utterance_ids()
    corpus = parser.transform(corpus)
    corpus = ps.transform(corpus, markers=True)
    print('--------prediciting politeness-----------')
    pred = clf.transform(corpus)
    df_pred_polite = clf.summarize(pred)
    polite_num = sum(df_pred_polite['prediction'])
    # creating df with neutral utterance
    df_pred_nuetral = df_pred_polite[df_pred_polite['prediction'] == 0]
    nuetral_num = len(df_pred_nuetral)
    return [polite_num, nuetral_num]

'''
###################################
assign the readability score to the answers in the qna dataframe using the flesch kincaid method. 

params: 
df - pd.DataFrame object that stores the questions and the answers

return:
a list of readability scores that assig to each of the sentence in the answer column 
###################################
'''      
        
def get_readability_score(df):
    score =  []
    for i in tqdm(range(len(df))):
        try:
            r = Readability(df['answer'][i]).flesch()
            score.append(r.score) 
        except:
            continue
    return score      
        
'''
categorize the score into different grade level. 
'''

def categorize_score(score):
    if 90.0 <= score <= 100.0:
        return "5th grade"
    elif 80.0 <= score < 90.0:
        return "6th grade"
    elif 70.0 <= score < 80.0:
        return "7th grade"
    elif 60.0 <= score < 70.0:
        return "8th & 9th grade"
    elif 50.0 <= score < 60.0:
        return "10th to 12th grade"
    elif 30.0 <= score < 50.0:
        return "College"
    elif 10.0 <= score < 30.0:
        return "College graduate"
    elif 0.0 <= score < 10.0:
        return "Professional"
    else:
        return "Invalid Score"
        
'''
create the readability score plot using the readability df we produced. 
'''        
def create_readability_plot(df, model_name:str):
    data = {
        "Grade Level": df[df['model']==model_name]['grade'],
        "Question Dialect": df[df['model']==model_name]['question'],
        "Frequency": df[df['model']==model_name]['frequency'] # Notice one value is 0
    }
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=df,
        x="Grade Level",
        y="Frequency",
        hue="Question Dialect",
        palette="muted"
    )
    
    # Adding values on top of the bars
    for p in ax.patches:
        if p.get_height() > 0:  # Only add text for bars with height > 0
            ax.annotate(
                f'{p.get_height():.0f}',  # Format the value as an integer
                (p.get_x() + p.get_width() / 2., p.get_height()),  # Position at bar center
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                xytext=(0, 8),  # Offset text position by 8 points
                textcoords='offset points'
            )
    
    # Adding labels
    plt.title(f"Distribution Flesch-Kincaid Readability Grade Level Equivalent of the Explanations({model_name})", fontsize=16)
    plt.xlabel("Grade Level", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation = 25)
    
    # Show the plot
    # plt.savefig("flesch_kincaid_readability_grade_gemma2.png", dpi=300, format='png', bbox_inches='tight')
    plt.legend(title="Question Dialect")
    plt.tight_layout()
    plt.show()



'''
calculates entropy 

params: data - list of elements.
'''

def calculate_entropy(data):
    # Count the frequency of each unique element
    counts = Counter(data)
    
    # Total number of elements
    total_elements = len(data)
    
    # Calculate the entropy
    entropy = 0
    for count in counts.values():
        # Calculate the probability of each unique element
        p = count / total_elements
        # Apply the entropy formula: -p * log2(p)
        entropy -= p * math.log2(p)
    
    return entropy

def get_avg_length(df1, df2 ):
    total_token1 = 0
    total_token2 = 0
    df_token1 = []
    df_token2 = []
    for i in range(len(df1)):
        total_token1 +=len(df1['answer'][i].split())
        df_token1.append(len(df1['answer'][i].split()))
    for i in range(len(df2)):
        total_token2 +=len(df2['answer'][i].split())
        df_token2.append(len(df2['answer'][i].split()))
    avg1 = total_token1/len(df1)
    avg2 = total_token2/len(df2)
    t_statistic, p_value = stats.ttest_ind(df_token1, df_token2)
    return avg1, avg2, t_statistic, p_value



'''
creates dataframe for linguistic marker

params: df-  pd.DataFrame object, the dataframe that you need to extract linguistic marker from
model - string, the name of the model that produced the explanation (eg, llama3.1, gemma2, mistral etc)
dialect - AAE (african american english) or SAE (standard american english)

'''
    
def create_ling_marker_df (df, model, dialect):
    merged_counter_sae = Counter()
    columns = ['ppron','i','you','we','they','social','posemo','negemo','tentat','certain','percept'] 
    token_num_sae = 0
    temp_num_sae = 0
    temp_merged_counter_sae = Counter()
    df_count = pd.DataFrame(columns = columns)
    for i in range(len(df)):
        answer  = df.loc[i]['answer']
        count_tokens = tokenize(answer)
        for tok in count_tokens:
            token_num_sae+=1
            temp_num_sae+=1
        answer_tokens = tokenize(answer)
        if temp_num_sae >=1000:
            temp_normalized_counter_sae = Counter({word: (count / temp_num_sae)*1000 for word, count in temp_merged_counter_sae.items()})
            temp_count = []
            for i in columns:
                temp_count.append(temp_normalized_counter_sae[i])
            df_count.loc[len(df_count)] = temp_count
            temp_num_sae = 0
            temp_merged_counter_sae = Counter()
    
        # now flatmap over all the categories in all of the tokens using a generator:
        answer_counts = Counter(category for token in answer_tokens for category in parse(token))
        # and print the results:
        merged_counter_sae +=answer_counts
        temp_merged_counter_sae +=answer_counts
    normalized_counter_sae = Counter({word: (count / token_num_sae)*1000 for word, count in merged_counter_sae.items()})
    ling_count = [model, dialect]
    for i in columns:
        ling_count.append(normalized_counter_sae[i])
    return df_count
        
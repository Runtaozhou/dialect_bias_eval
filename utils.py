'''
Util functions
'''

import re
import os
import time
import pandas as pd
import evaluate 
import random
import datasets
from multivalue import Dialects
from pydantic import BaseModel, validator
from llm_chain import create_chain

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
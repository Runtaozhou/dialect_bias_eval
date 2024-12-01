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


def format_docstring(docstring: str) -> str:
    """Format a docstring for use in a prompt template."""
    return re.sub("\n +", "\n", docstring).strip()

'''
################################################################
converter that convert SAE to AAVE
params: 
    - sentences: str the sae sentence that you want to convert to AAVE
    - aal_phonate: class object from AALPhonate that is need for phonate conversion.
    - converter_type: str, the method that you want to use for conversion: "multi_value", "phonate", "both"
    IMPORTANT NOTE: the llm conversion in our experiment uses converter_type: "llm_rulebased_persona"
################################################################
'''
def aave_converter(sentence, aal_phonate, converter_type = "multi_value"):
        if converter_type == "multi_value":
            try:
                aave = Dialects.AfricanAmericanVernacular()
                converted = aave.transform(sentence)
            except:
                converted = sentence
        elif converter_type == "phonate":
            # aal_phonate = AALPhonate(config = 'default_config.json')
            aal_phonate.update_probs(1.0)
            _, _, _, clean_out = aal_phonate.full_phon_aug([sentence])
            converted = clean_out[0]
        elif converter_type == "both":
            aave = Dialects.AfricanAmericanVernacular()
            # aal_phonate = AALPhonate(config = 'default_config.json')
            aal_phonate.update_probs(1.0)
            try:
                converted_multi_value = aave.transform(sentence)
                _, _, _, clean_out = aal_phonate.full_phon_aug([converted_multi_value])
                converted = clean_out[0]
            except:
                converted = sentence
        elif converter_type == "llm_persona":
            print("converting the sentence using gpt")
            converted = aae_persona('gpt-3.5', sentence)
            if converted[0] == '"':
                converted = converted[1:-1]
        elif converter_type == "llm_rulebased":
            print("converting the sentence using informed gpt")
            converted = aae_rulebased('gpt-3.5', sentence)
            if converted[0] == '"':
                converted = converted[1:-1]
        elif converter_type == "llm_rulebased_persona":
            converted = aae_rulebased_persona('gpt-3.5', sentence)
            if converted[0] == '"':
                converted = converted[1:-1]
        # no change         
        else:
            converted = sentence
        return converted 
    
def replace_words(text, replacement_dict):
    # Sort dictionary by length of keys to avoid partial matches (e.g., 'dis' vs 'distance')
    sorted_dict = dict(sorted(replacement_dict.items(), key=lambda x: -len(x[0])))
    
    # Replace each word in the dictionary using re.sub
    for word, replacement in sorted_dict.items():
        # \b ensures the word boundary is respected
        text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text)
    
    return text

def aae_rulebased(model_name, sentence):
    rules = "1. Null copula: Verbal copula is deleted (e.g., “he a delivery man” → “he’s a delivery man”).\
        2. Negative concord: Negatives agree with each other (e.g., “nobody never say nothing” → “nobody ever says anything”).\
        3. Negative inversion: Auxiliary raises, but interpretation remains the same (e.g., “don’t nobody never say nothing to them” → “nobody ever says anything to them”).\
        4. Deletion of possessive /s/: Possessive markers are omitted, but the meaning remains (e.g., “his baby mama brother friend was there” → “his baby’s mother’s brother’s friend was there”).\
        5. Habitual 'be like': descrbing something (e.g., “This song be like fire” → “This song is amazing”).\
        6. Stressed 'been': short for have been doing something (e.g., “I been working out” → “I have been working out”).\
        7. Preterite 'had': Signals the preterite or past action (e.g., “we had went to the store” → “we went to the store”).\
        8. Question inversion in subordinate clauses: Inverts clauses similar to Standard English (e.g., “I was wondering did his white friend call?” → “I was wondering whether his white friend called”).\
        9. Reduction of negation: Contraction of auxiliary verbs and negation (e.g., “I ain’t eem be feeling that” → “I don’t much care for that”).\
        10. Quotative 'talkin’ 'bout': Used as a verb of quotation (e.g., “she talkin’ 'bout he don’t live here no more” → “she’s saying he doesn’t live here anymore”).\
        11. Modal 'tryna': Short form of “trying to” (e.g., “I was tryna go to the store” → “I was planning on going to the store”).\
        12. Expletive 'it': Used in place of “there” (e.g., “it’s a lot of money out there” → “there’s a lot of money out there”).\
        13. Shortening ing words to in': anyword ends with ing has to be converted into a in' format (e.g. “he is playing” → “he is playin'”)"

    answer_list = []
    counter = 0 
    # Define the prompt template for multiple-choice questions
    template = "Please translate the following sentence: '{sentence}' using the 13 translation rules provided as references: {rules}. Your output must follow these guidelines: \
    1. Only provide the translation. Do not mention or explain how the translation was done. \
    2. Do not mention any of the 13 rules in your translation.\
    3. Format the output exactly like this: 'The translation is: ...' \
    4. Ensure the sentence sounds natural and realistic in AAVE. "
    input_variables = ["sentence", "rules"]
    #create chain
    chain = create_chain(model_name, template, input_variables)
    input_data = {"sentence": sentence,
                 "rules": rules}
    # invoke the questions
    response = chain.invoke(input_data)
    text = response['text']
    pattern = r"The translation is:\s*(.*)"
    match = re.search(pattern, text)
    if match:
        # Extract the sentence that follows "The translated sentence is:"
        translated_sentence = match.group(1)
        # print(f"The sentence after 'The translated sentence is:' is: {translated_sentence}")
    else:
        translated_sentence = sentence
        # print("No match found.")
    # last step manual convert: 
    updated_translation = replace_words(translated_sentence,replacement_dict)
    return updated_translation



def aae_persona(model_name, sentence):
    answer_list = []
    counter = 0 
    # Define the prompt template for multiple-choice questions
    template = "You are an African American speaker fluent in both SAE (Standard American English) and AAVE (African American Vernacular English). I need your help translating the following sentence from SAE to AAVE in a way that feels natural and preserves the original meaning and tone. Avoid overly formal language, and aim for authenticity in the AAVE translation. Your output must follow these guidelines: \
    1. Only provide the translation. Do not mention or explain how the translation was done. \
    2. Do not mention any of the 13 rules in your translation.\
    3. Format the output exactly like this: 'The translation is: ...' \
    4. Ensure the sentence sounds natural and realistic in AAVE. "
    input_variables = ["sentence", "rules"]
    #create chain
    chain = create_chain(model_name, template, input_variables)
    input_data = {"sentence": sentence}
    # invoke the questions
    response = chain.invoke(input_data)
    text = response['text']
    pattern = r"The translation is:\s*(.*)"
    match = re.search(pattern, text)
    if match:
        # Extract the sentence that follows "The translated sentence is:"
        translated_sentence = match.group(1)
        # print(f"The sentence after 'The translated sentence is:' is: {translated_sentence}")
    else:
        translated_sentence = sentence
        # print("No match found.")
    # last step manual convert: 
    return translated_sentence


def aae_rulebased_persona(model_name, sentence):
    # Sample dictionary for replacements
    replacement_dict = {
        "isn't": "ain't",
        "going to": "gonna",
        "because": "cuz",
        "have" : "got",
        "ing": "in",
        "about": "bout"
        
    }
    rules = "1. Null copula: Verbal copula is deleted (e.g., “he a delivery man” → “he’s a delivery man”).\
        2. Negative concord: Negatives agree with each other (e.g., “nobody never say nothing” → “nobody ever says anything”).\
        3. Negative inversion: Auxiliary raises, but interpretation remains the same (e.g., “don’t nobody never say nothing to them” → “nobody ever says anything to them”).\
        4. Deletion of possessive /s/: Possessive markers are omitted, but the meaning remains (e.g., “his baby mama brother friend was there” → “his baby’s mother’s brother’s friend was there”).\
        5. Habitual 'be like': descrbing something (e.g., “This song be like fire” → “This song is amazing”).\
        6. Stressed 'been': short for have been doing something (e.g., “I been working out” → “I have been working out”).\
        7. Preterite 'had': Signals the preterite or past action (e.g., “we had went to the store” → “we went to the store”).\
        8. Question inversion in subordinate clauses: Inverts clauses similar to Standard English (e.g., “I was wondering did his white friend call?” → “I was wondering whether his white friend called”).\
        9. Reduction of negation: Contraction of auxiliary verbs and negation (e.g., “I ain’t eem be feeling that” → “I don’t much care for that”).\
        10. Quotative 'talkin’ 'bout': Used as a verb of quotation (e.g., “she talkin’ 'bout he don’t live here no more” → “she’s saying he doesn’t live here anymore”).\
        11. Modal 'tryna': Short form of “trying to” (e.g., “I was tryna go to the store” → “I was planning on going to the store”).\
        12. Expletive 'it': Used in place of “there” (e.g., “it’s a lot of money out there” → “there’s a lot of money out there”).\
        13. Shortening ing words to in': anyword ends with ing has to be converted into a in' format (e.g. “he is playing” → “he is playin'”)"

    answer_list = []
    counter = 0 
    # Define the prompt template for multiple-choice questions
    template = "You are an African American speaker fluent in both SAE (Standard American English) and AAVE (African American Vernacular English). I need your help translating the following sentence from SAE to AAVE in a way that feels natural and preserves the original meaning and tone. Avoid overly formal language, and aim for authenticity in the AAVE translation. Please translate the following sentence: '{sentence}' using the 13 translation rules provided as references: {rules}. Your output must follow these guidelines: \
    1. Only provide the translation. Do not mention or explain how the translation was done. \
    2. Do not mention any of the 13 rules in your translation.\
    3. Format the output exactly like this: 'The translation is: ...' \
    4. Ensure the sentence sounds natural and realistic in AAVE. "
    input_variables = ["sentence", "rules"]
    #create chain
    chain = create_chain(model_name, template, input_variables)
    input_data = {"sentence": sentence,
                 "rules": rules}
    # invoke the questions
    response = chain.invoke(input_data)
    text = response['text']
    pattern = r"The translation is:\s*(.*)"
    match = re.search(pattern, text)
    if match:
        # Extract the sentence that follows "The translated sentence is:"
        translated_sentence = match.group(1)
        # print(f"The sentence after 'The translated sentence is:' is: {translated_sentence}")
    else:
        translated_sentence = sentence
        # print("No match found.")
    # last step manual convert: 
    updated_translation = replace_words(translated_sentence,replacement_dict)
    return updated_translation
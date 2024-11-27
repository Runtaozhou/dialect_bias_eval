import re
import warnings
import spacy
import time
import pandas as pd
from tqdm import tqdm
from dataset import get_dataset
from pydantic import BaseModel
from typing import Optional
from llm_chain import create_chain
from PhonATe.phonate import AALPhonate, phonate_filter
from pydantic import BaseModel, ValidationError
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
from aave_converter import aave_converter
# Load a spaCy model
# nlp = spacy.load("en_core_web_sm")
from multivalue import Dialects

warnings.simplefilter(action='ignore', category=FutureWarning)

# Define a Pydantic model for the output format
class QuestionResponse(BaseModel):
    question: str  # Only the generated question


class ChoiceResponse(BaseModel):
    choice: str  # Should be "A", "B", "C", or "D

'''
################################################################
generator that generate the question prompts

params: 
    - dataset_name: one specific subject in MMLU benchmark
    - converter_type: str indicates which AAVE converter you want to use ("phonate", "llm", "multi_value" and "both"). Only used when aave == True
    - aal_phonate: class object from AALPhonate that is need for phonate conversion. Only used when aave == True
    -aave_instruct: bool indicates if we only want to change the instruction part of the question prompt to AAVE and keep the question as SAE
    -aave: bool indicates if we want to convert the whole question prompt from SAE to AAVE.


In function generate:
    - n_run: int , indicates how many questions you want to simulate from a given subject. 
################################################################
'''

class mmlu_question_generator:
    def __init__(self, dataset_name, converter_type, aal_phonate, aave_instruct = False, aave = True):
        self.dataset_name = dataset_name
        self.model_name = "gpt-3.5"
        self.dataset =  get_dataset(dataset_name).get_data()[:]
        self.aal_phonate = aal_phonate
        self.converter_type  = converter_type
        self.aave_instruct = aave_instruct
        self.aave = aave
        self.question_list = []
        self.pure_question_list = []
        self.answer_list = []
        self.subject_list= []
    def generate(self, n_run):
        for i in range(n_run):
            # Assuming you iterate over questions in the dataset
            question_text = self.dataset_name.split('-')[1], self.dataset[i]['question']  # Using the first question as an example
            self.pure_question_list.append(question_text[1])
            self.subject_list.append(question_text[0])
            answer_text = self.dataset_name.split('-')[1], self.dataset[i]['answer']
            self.answer_list.append(answer_text[1])
            # experiment for only changing the instruction but keep the question the same. 
            if self.aave_instruct ==True:
                text = f"Aye fam, I'm stuck on this multiple choice question: {question_text[1]}. Which one I'm posed to pick? Hook me up with some clues or sum"
            else:
                # Define the prompt template for multiple-choice questions
                template = (
                    "You are stuck with a multiple choice question: {question}. You would like to ask a "
                    "Large Language Model for help. Please generate only the question you would ask the "
                    "Large Language Model, The question needs to include the original multiple choice question and the 4 options"
                    "Please make sure that you pretend to be a human and the question should sound as realistic as possible"
                )
                input_variables = ["question"]
            
                # Create the chain
                chain = create_chain(self.model_name, template, input_variables)
                input_data = {"question": question_text[1]}
                
                # Invoke the question generation
                response = chain.invoke(input_data)
                text = response['text']
            
            # Parse the output with the Pydantic model to extract only the question
            try:
                llm_response = QuestionResponse.parse_raw(text)  # Assuming response text is JSON formatted  # Print only the question
            except Exception as e:
                llm_response = None
            question_final = llm_response.question if llm_response else text
            if self.aave == True:
                print("aave_converting")
                question_final = aave_converter(question_final ,aal_phonate = aal_phonate,converter_type = converter_type )
            self.question_list.append(question_final)
        return self.question_list, self.pure_question_list, self.answer_list, self.subject_list

'''
################################################################
generator that generate the answers to the question prompts

params: 
    - model_name: name of the model that you want to use to generate your answer. 
    - question_lst: list of questions that you want to ask the LLMs to give answer to. (should come directly from question generator.)
################################################################
'''

class answer_generator:
    def __init__(self, model_name, question_lst):
        self.model_name = model_name
        self.question_lst = question_lst
        self.answer_lst = []
    def generate(self):
        for question in self.question_lst:
            # Define the prompt template for multiple-choice questions
            template = (
                "Someone asked you a multiple choice question: {question}, Please first provide an detailed explaination then your final answer"
                "You need to make your explaination sounds as natural and realistic as possible"
                "At the end, you should state the letter option (A, B, C, or D) you choose."
                "You answer should strictly be less than 400 words."
            )
            input_variables = ["question"]
        
            # Create the chain
            chain = create_chain(self.model_name , template, input_variables)
            input_data = {"question": question}
            
            # Invoke the question generation
            response = chain.invoke(input_data)
            text = response['text']
            self.answer_lst.append(text)
        return self.answer_lst

'''
################################################################
extractor that extract the letter answers from an answer with explanations

params: 
    - answer_lst: list of answers coming from answer_generator that you actually want to extract the letter answer with. 
################################################################
'''

class answer_extractor:
    def __init__(self, answer_lst):
        self.answer_lst = answer_lst
        self.letter_answer_list = []
    def generate(self):
        model_name = "gpt-3.5"
        for text in self.answer_lst:
            # Define a prompt to ask the LLM to only output the letter choice
            prompt_template = (
                "Given the following text:\n\n'{text}'\n\nIdentify the answer choice (A, B, C, or D) "
                "from the text and return only the letter. Do not include any additional text."
            )
            input_variables = ["text"]
        
            # Create the chain with the prompt
            chain = create_chain(model_name, prompt_template, input_variables)
            input_data = {"text": text}
            
            # Invoke the LLM to generate the answer choice
            response = chain.invoke(input_data)
            extracted_text = response['text'].strip()
        
            # Try to parse the output with Pydantic; if parsing fails, attempt regex extraction
            try:
                choice_response = ChoiceResponse(choice=extracted_text)  # Direct parse attempt
            except ValidationError:
                print("Parsing failed, attempting regex extraction.")
                # Fallback: Use regex to find a single letter A-D in parentheses
                match = re.search(r"\((A|B|C|D)\)", text)
                if match:
                    choice_response = ChoiceResponse(choice=match.group(1))
                    print(choice_response.choice)
                else:
                    choice_response = None
            if choice_response:
                choice = choice_response.choice  
            else:
                choice = "N/A"
            self.letter_answer_list.append(choice)
    
        return self.letter_answer_list

'''
################################################################
Running a single simulation cycle for questions in one specific subject in MMLU

params: 
    - model_name: name of the model that you want to use to generate your answer. 
    - dataset_name: one subject in MMLU benchmark
    - aal_phonate: class object from AALPhonate that is need for phonate conversion. Only used when aave == True
    - n_run: int , indicates how many questions you want to simulate from a given subject. 
    - aave: bool indicates if we want to convert the whole question prompt from SAE to AAVE.
    -aave_instruct: bool indicates if we only want to change the instruction part of the question prompt to AAVE and keep the question as SAE
    - converter_type: str indicates which AAVE converter you want to use ("phonate", "llm", "multi_value" and "both"). Only used when aave == True
    
################################################################
'''

def simulate_one_question (model_name, dataset_name, aal_phonate, n_run, aave, aave_instruct, converter_type):
    q_generator = mmlu_question_generator(dataset_name = dataset_name, 
                                            aal_phonate = aal_phonate, 
                                            aave_instruct = aave_instruct,
                                            aave = aave,
                                            converter_type = converter_type)
    question_list, pure_question_list, correct_answer_list, subject_list = q_generator.generate(n_run = n_run)
    a_generator = answer_generator(model_name = model_name, question_lst = question_list)
    answer_list = a_generator.generate()
    a_extractor = answer_extractor(answer_lst = answer_list)
    letter_answer_list = a_extractor.generate()
    return question_list,answer_list,letter_answer_list, pure_question_list,correct_answer_list, subject_list


'''
################################################################
Running all the simulations for questions in all the subjects specified in dataset_names

params: 
    - dataset_names: list of subjects in MMLU benchmark
    - model_name: name of the model that you want to use to generate your answer. 
    - aave: bool indicates if we want to convert the whole question prompt from SAE to AAVE.
    - n_run: int , indicates how many questions you want to simulate from a given subject. 
    - converter_type: str indicates which AAVE converter you want to use ("phonate", "llm", "multi_value" and "both"). Only used when aave == True
    -aave_instruct: bool indicates if we only want to change the instruction part of the question prompt to AAVE and keep the question as SAE
################################################################
'''
    
def run_simulation(dataset_names, model_name, aave, n_run, aave_instruct, converter_type):
    aal_phonate = AALPhonate(config = 'default_config.json')
    total_question_list = []
    total_answer_list = []
    total_letter_answer_list = []
    total_pure_question_list = []
    total_correct_answer_list = []
    total_subject_list = []
    for subject in tqdm(dataset_names):
        try:
            question_list, answer_list , letter_answer_list, pure_question_list, correct_answer_list, subject_list = simulate_one_question(                                                          model_name = model_name, 
                                                                                subject, 
                                                                                aave = aave, 
                                                                                aave_instruct = aave_instruct,
                                                                                aal_phonate = aal_phonate,
                                                                                n_run = n_run, 
                                                                                converter_type = converter_type)
            total_question_list.extend(question_list)
            total_answer_list.extend(answer_list)
            total_letter_answer_list.extend(letter_answer_list)
            total_correct_answer_list.extend(correct_answer_list)
            total_pure_question_list.extend(pure_question_list)
            total_subject_list.extend(subject_list)   
        except:
            continue
    df = pd.DataFrame(data = {'subject':total_subject_list, 
                                   'question': total_question_list, 
                                   'answer':total_answer_list, 
                                   'letter_answer':total_letter_answer_list, 
                                   'pure_question': total_pure_question_list, 
                                   'correct_answer':total_correct_answer_list})
    return df
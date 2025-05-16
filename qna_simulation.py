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

# Define prompt strategies
PROMPT_STRATEGIES = {
    "expert_teacher": (
        "You are an experienced teacher with expertise in working with students from diverse linguistic backgrounds. "
        "Your goal is to explain concepts clearly while respecting and accommodating different language varieties. "
        "Please read the following question and provide your response: {question}"
    ),
    "cultural_context": (
        "The following question may be presented in different language varieties, including African American English (AAE) "
        "or Standard American English (SAE). Each variety has its own valid grammatical rules and cultural context. "
        "Please analyze the question with respect to its linguistic context: {question}"
    ),
    "dialect_recognition": (
        "When responding to this question, be aware that it may be expressed in different English dialects. "
        "Apply your understanding of dialect-specific features and grammatical patterns. "
        "Consider all dialectal variations as equally valid forms of expression: {question}"
    ),
    "readability_focus": (
        "Please ensure your response is clear and accessible across different English varieties. "
        "Focus on maintaining consistent meaning and comprehension regardless of the dialect used. "
        "Analyze the following question: {question}"
    ),
    "multi_strategy": (
        "As an experienced educator skilled in working with diverse linguistic backgrounds, please address this question while:\n"
        "1. Recognizing and respecting different language varieties (including AAE and SAE)\n"
        "2. Ensuring clear communication across dialects\n"
        "3. Maintaining consistent comprehension\n"
        "4. Acknowledging the validity of different grammatical patterns\n\n"
        "Question: {question}"
    ),
    "default": (
        "You are stuck with a multiple choice question: {question}. You would like to ask a "
        "Large Language Model for help. Please generate only the question you would ask the "
        "Large Language Model, The question needs to include the original multiple choice question and the 4 options"
        "Please make sure that you pretend to be a human and the question should sound as realistic as possible"
    )
}

# Define a Pydantic model for the output format
class QuestionResponse(BaseModel):
    question: str  # Only the generated question


class ChoiceResponse(BaseModel):
    choice: str  # Should be "A", "B", "C", or "D

'''
################################################################
generator that generate the question prompts

params: 
    - dataset_name: str, name of dataset. available dataset: "mmlu", "bigbench"
    - category_name: str, one specific category/subject in the benchmark dataset 
    - converter_type: str, indicates which AAVE converter you want to use ("phonate", "llm", "multi_value" and "both"). Only used when aave == True
    - aal_phonate: class object from AALPhonate that is need for phonate conversion. Only used when aave == True
    -aave_instruct: bool indicates if we only want to change the instruction part of the question prompt to AAVE and keep the question as SAE
    -aave: bool indicates if we want to convert the whole question prompt from SAE to AAVE.


In function generate:
    - n_run: int , indicates how many questions you want to simulate from a given subject. 
################################################################
'''

class question_generator:
    def __init__(self, dataset_name, category_name, converter_type, aal_phonate, aave_instruct=False, aave=True, prompt_strategy="default"):
        self.dataset_name = dataset_name
        self.category_name = category_name
        self.model_name = "gpt-3.5"
        if dataset_name == "mmlu":
            self.dataset =  get_dataset(category_name).get_data()[:]
        elif dataset_name == "bigbench":
            bigbench_df = pd.read_csv('bigbench_hard.csv')
            self.dataset = bigbench_df[bigbench_df['category'] == category_name].reset_index(drop=True)
        self.aal_phonate = aal_phonate
        self.converter_type  = converter_type
        self.aave_instruct = aave_instruct
        self.aave = aave
        self.prompt_strategy = prompt_strategy
        self.length = len(self.dataset)
        self.question_list = []
        self.pure_question_list = []
        self.answer_list = []
        self.subject_list= []
    def generate(self, n_run):
        print(self.dataset)
        # special case where you want to run every questions in your benchmark dataset. 
        if n_run == -1:
            n_run = self.length
            print(f'loading {n_run} questions!')
        for i in range(n_run):
            if self.dataset_name == "mmlu":
                question_text =  self.dataset[i]['question']  
                subject = self.category_name.split('-')[1]
                self.pure_question_list.append(question_text)
                self.subject_list.append(subject)
                answer_text = self.dataset[i]['answer']
                self.answer_list.append(answer_text)
            elif self.dataset_name == "bigbench":
                question_text = self.dataset.loc[i]['question']  
                self.pure_question_list.append(question_text)
                subject = self.dataset.loc[i]['category']  
                self.subject_list.append(subject)
                answer_text = self.dataset.loc[i]['answer']  
                self.answer_list.append(answer_text)
            # experiment for only changing the instruction but keep the question the same. 
            if self.aave_instruct ==True:
                text = f"Aye fam, I'm stuck on this multiple choice question: {question_text}. Which one I'm posed to pick? Hook me up with some clues or sum"
            else:
                # Use the selected prompt strategy
                template = PROMPT_STRATEGIES.get(self.prompt_strategy, PROMPT_STRATEGIES["default"])
                input_variables = ["question"]
            
                # Create the chain
                chain = create_chain(self.model_name, template, input_variables)
                input_data = {"question": question_text}
                
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
                question_final = aave_converter(question_final ,aal_phonate = self.aal_phonate,converter_type = self.converter_type )
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
                "At the end, you should state the letter option (A, B, C, D, E or F) you choose."
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

class explanation_generator:
    def __init__(self, model_name, question_lst, answer_lst):
        self.model_name = model_name
        self.question_lst = question_lst
        self.answer_lst = answer_lst
        self.explanation_lst = []
    
    def generate(self):
        for question, answer in zip(self.question_lst, self.answer_lst):
            # Define the prompt template for explanation
            template = (
                "For the following question and answer:\n"
                "Question: {question}\n"
                "Answer: {answer}\n\n"
                "Please provide a detailed explanation of why this answer is correct. "
                "Include the reasoning process, relevant concepts, and any assumptions made. "
                "Your explanation should be clear and educational, helping someone understand "
                "not just what the answer is, but why it's correct."
            )
            input_variables = ["question", "answer"]
        
            # Create the chain
            chain = create_chain(self.model_name, template, input_variables)
            input_data = {
                "question": question,
                "answer": answer
            }
            
            # Invoke the explanation generation
            response = chain.invoke(input_data)
            text = response['text']
            self.explanation_lst.append(text)
        return self.explanation_lst

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
                "Given the following text:\n\n'{text}'\n\nIdentify the answer choice (A, B, C, D, E or F) "
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
                match = re.search(r"\((A|B|C|D|E|F)\)", text)
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
    - category_name: one subject in MMLU benchmark
    - aal_phonate: class object from AALPhonate that is need for phonate conversion. Only used when aave == True
    - n_run: int , indicates how many questions you want to simulate from a given subject. 
    - aave: bool indicates if we want to convert the whole question prompt from SAE to AAVE.
    -aave_instruct: bool indicates if we only want to change the instruction part of the question prompt to AAVE and keep the question as SAE
    - converter_type: str indicates which AAVE converter you want to use ("phonate", "llm", "multi_value" and "both"). Only used when aave == True
    
################################################################
'''

def simulate_one_question(model_name, dataset_name, category_name, aal_phonate, n_run, aave, aave_instruct, converter_type, prompt_strategy="default", get_explanation=False):
    q_generator = question_generator(
        dataset_name=dataset_name,
        category_name=category_name, 
        aal_phonate=aal_phonate, 
        aave_instruct=aave_instruct,
        aave=aave,
        converter_type=converter_type,
        prompt_strategy=prompt_strategy
    )
    question_list, pure_question_list, correct_answer_list, subject_list = q_generator.generate(n_run=n_run)
    a_generator = answer_generator(model_name=model_name, question_lst=question_list)
    answer_list = a_generator.generate()
    a_extractor = answer_extractor(answer_lst=answer_list)
    letter_answer_list = a_extractor.generate()
    
    explanation_list = []
    if get_explanation:
        e_generator = explanation_generator(model_name=model_name, question_lst=question_list, answer_lst=answer_list)
        explanation_list = e_generator.generate()
    
    return question_list, answer_list, letter_answer_list, pure_question_list, correct_answer_list, subject_list, explanation_list


'''
################################################################
Running all the simulations for questions in all the subjects specified in category_names

params: 
    - category_names: list of subjects in MMLU benchmark
    - model_name: name of the model that you want to use to generate your answer. 
    - aave: bool indicates if we want to convert the whole question prompt from SAE to AAVE.
    - n_run: int , indicates how many questions you want to simulate from a given subject. 
    - converter_type: str indicates which AAVE converter you want to use ("phonate", "llm", "multi_value" and "both"). Only used when aave == True
    -aave_instruct: bool indicates if we only want to change the instruction part of the question prompt to AAVE and keep the question as SAE
################################################################
'''
    
def run_simulation(dataset_name, category_names, model_name, aave, n_run, aave_instruct, converter_type, prompt_strategy="default", get_explanation=False):
    aal_phonate = AALPhonate(config='default_config.json')
    total_question_list = []
    total_answer_list = []
    total_letter_answer_list = []
    total_pure_question_list = []
    total_correct_answer_list = []
    total_subject_list = []
    total_explanation_list = []
    
    for subject in tqdm(category_names):
        try:
            question_list, answer_list, letter_answer_list, pure_question_list, correct_answer_list, subject_list, explanation_list = simulate_one_question(
                dataset_name=dataset_name,
                model_name=model_name, 
                category_name=subject, 
                aave=aave, 
                aave_instruct=aave_instruct,
                aal_phonate=aal_phonate,
                n_run=n_run, 
                converter_type=converter_type,
                prompt_strategy=prompt_strategy,
                get_explanation=get_explanation
            )
            total_question_list.extend(question_list)
            total_answer_list.extend(answer_list)
            total_letter_answer_list.extend(letter_answer_list)
            total_correct_answer_list.extend(correct_answer_list)
            total_pure_question_list.extend(pure_question_list)
            total_subject_list.extend(subject_list)
            if get_explanation:
                total_explanation_list.extend(explanation_list)
        except:
            print('encountered_troubles')
            continue
    
    # Create DataFrame with or without explanation column
    df_data = {
        'subject': total_subject_list, 
        'question': total_question_list, 
        'answer': total_answer_list, 
        'letter_answer': total_letter_answer_list, 
        'pure_question': total_pure_question_list, 
        'correct_answer': total_correct_answer_list
    }
    
    if get_explanation:
        df_data['explanation'] = total_explanation_list
    
    df = pd.DataFrame(data=df_data)
    return df
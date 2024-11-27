import os
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI

os.environ["REDIS_OM_URL"] = "redis://:@localhost:6379"
os.environ["OPENAI_API_KEY_3"] =  "0626b133c7b5407d87aa8b93f3331031"
os.environ["OPENAI_API_KEY_4"] =  "bffeba6e73e24113bf6cd0457b0360f3"
os.environ["TOGETHER_API_KEY"] = "3dafbeb1fa9abba4c743b2529e18654de77fe912a3fb5a35a52985da520c0ea5"

'''
################################################################
creating LLMChain. 
params: 
    - model name: str the name LLM model ('gpt-3.5', 'gpt-4', 'llama3.1', etc)
    - template: HumanMessagePromptTemplate from langchain.prompts that gives LLM a prompt template 
    - input_variables: variables that will be feed into the prompt template. More details please refer to qna_simulation.mmlu_question_generator.
################################################################
'''

def create_chain(model_name, template, input_variables):
    if "gpt-3.5" in model_name:
        chat = AzureChatOpenAI(
            openai_api_version="2023-07-01-preview",
            openai_api_key=os.environ.get("OPENAI_API_KEY_3"),
            azure_endpoint="https://rtp2-gpt35.openai.azure.com/",
            model_name="gpt-35-turbo",
            temperature=0.9
        )
    elif "gpt-4" in model_name:
        chat = AzureChatOpenAI(
            openai_api_version="2023-07-01-preview",
            openai_api_key=os.environ.get("OPENAI_API_KEY_4"),
            azure_endpoint="https://rtp2-shared.openai.azure.com/",
            model_name="gpt-4-turbo",
            temperature=0.9
        )
    ## adding LLM models via chatollama
    elif "llama3.1" in model_name:
        chat = ChatOllama(model="llama3.1", base_url="http://127.0.0.1:11434")
    elif "llama3.2" in model_name:
        chat = ChatOllama(model="llama3.2:3b", base_url="http://127.0.0.1:11434")
    elif "qwen2.5" in model_name:
        chat = ChatOllama(model="qwen2.5", base_url="http://127.0.0.1:11434")
    elif "gemma2" in model_name:
        chat = ChatOllama(model="gemma2", base_url="http://127.0.0.1:11434")
    elif "phi3.5" in model_name:
        chat = ChatOllama(model="phi3.5", base_url="http://127.0.0.1:11434")
    elif "phi3" in model_name:
        chat = ChatOllama(model="phi3", base_url="http://127.0.0.1:11434")
    elif "mistral" in model_name:
        chat = ChatOllama(model="mistral", base_url="http://127.0.0.1:11434")
    else:
        raise ValueError("Model not supported")

    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(template=template, input_variables=input_variables)
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)
    return chain

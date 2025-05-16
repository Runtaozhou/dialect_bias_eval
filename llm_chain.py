import os
import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI

os.environ["REDIS_OM_URL"] = "redis://:@localhost:6379"
# os.environ["OPENAI_API_KEY_3"] =  "0626b133c7b5407d87aa8b93f333103"  1
# os.environ["OPENAI_API_KEY_4"] =  "bffeba6e73e24113bf6cd0457b0360f"  3
# os.environ["TOGETHER_API_KEY"] = "3dafbeb1fa9abba4c743b2529e18654de77fe912a3fb5a35a52985da520c0ea"  5

'''
################################################################
creating LLMChain. 
params: 
    - model name: str the name LLM model ('gpt-3.5', 'gpt-4', 'llama3.1', etc)
    - template: HumanMessagePromptTemplate from langchain.prompts that gives LLM a prompt template 
    - input_variables: variables that will be feed into the prompt template. More details please refer to qna_simulation.mmlu_question_generator.
################################################################
'''

class ParallelLLMChain:
    def __init__(self, model_name: str, template: str, input_variables: List[str], max_workers: int = 4):
        self.model_name = model_name
        self.template = template
        self.input_variables = input_variables
        self.max_workers = max_workers
        self.chains = self._create_chains()
        
    def _create_chains(self) -> List[LLMChain]:
        """Create multiple chains for parallel processing."""
        chains = []
        for _ in range(self.max_workers):
            if "gpt-3.5" in self.model_name:
                chat = AzureChatOpenAI(
                    openai_api_version="2023-07-01-preview",
                    openai_api_key=os.environ.get("OPENAI_API_KEY_3"),
                    azure_endpoint="https://rtp2-gpt35.openai.azure.com/",
                    model_name="gpt-35-turbo",
                    temperature=0.9
                )
            elif "gpt-4" in self.model_name:
                chat = AzureChatOpenAI(
                    openai_api_version="2023-07-01-preview",
                    openai_api_key=os.environ.get("OPENAI_API_KEY_4"),
                    azure_endpoint="https://rtp2-shared.openai.azure.com/",
                    model_name="gpt-4-turbo",
                    temperature=0.9
                )
            else:
                raise ValueError("Parallel processing only supported for GPT models")
            
            human_message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(template=self.template, input_variables=self.input_variables)
            )
            chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
            chain = LLMChain(llm=chat, prompt=chat_prompt_template)
            chains.append(chain)
        return chains
    
    def invoke_batch(self, input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple inputs in parallel."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a list of futures
            future_to_input = {
                executor.submit(self.chains[i % len(self.chains)].invoke, input_data): input_data
                for i, input_data in enumerate(input_data_list)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_input):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing input: {e}")
                    results.append({"text": f"Error: {str(e)}"})
        
        return results

def create_chain(model_name: str, template: str, input_variables: List[str], max_workers: Optional[int] = None) -> Any:
    """
    Create an LLM chain with optional parallel processing for GPT models.
    
    Args:
        model_name: Name of the LLM model
        template: Prompt template
        input_variables: List of input variables for the template
        max_workers: Number of parallel workers (only used for GPT models)
    
    Returns:
        Either a ParallelLLMChain (for GPT models with max_workers > 1) or a regular LLMChain
    """
    # For GPT models with parallel processing enabled
    if max_workers and max_workers > 1 and ("gpt-3.5" in model_name or "gpt-4" in model_name):
        return ParallelLLMChain(model_name, template, input_variables, max_workers)
    
    # For all other cases, use the original implementation
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

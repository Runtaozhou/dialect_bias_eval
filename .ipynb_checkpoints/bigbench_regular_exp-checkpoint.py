import os
import pickle
from qna_simulation import run_simulation
import pandas as pd

os.environ["REDIS_OM_URL"] = "redis://:@localhost:6379"
os.environ["OPENAI_API_KEY_3"] =  "0626b133c7b5407d87aa8b93f3331031"
os.environ["OPENAI_API_KEY_4"] =  "bffeba6e73e24113bf6cd0457b0360f3"
os.environ["TOGETHER_API_KEY"] = "3dafbeb1fa9abba4c743b2529e18654de77fe912a3fb5a35a52985da520c0ea5"


category_names = ['navigate', 'tracking_shuffled_objects_three_objects','temporal_sequences', 'date_understanding', 'penguins_in_a_table','causal_judgement']
model_name = "gpt-4"
print(model_name)
df_regular = run_simulation(dataset_name = "bigbench", category_names =category_names, model_name =model_name , aave= False, n_run = -1, aave_instruct = False, converter_type = "llm_rulebased_persona")
path = f"/home/uar6nw/Documents/LLM_Persona/dialect_bias_eval/bigbench_dataset/{model_name}/"
df_regular.to_csv(path+f'regular_bigbench_qna.csv',header= True, index = False)



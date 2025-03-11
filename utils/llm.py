import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from utils.constants import OPENAI_MODELS, GROQ_MODELS

def get_llm(model_version:str):

  if model_version in OPENAI_MODELS:
    llm = ChatOpenAI(api_key=os.environ.get('OPENAI_KEY'), 
                      model=model_version, 
                      temperature=.1)
    return llm
  elif model_version in GROQ_MODELS:
    llm = ChatGroq(api_key=os.environ.get('GROQ_KEY'), 
                    model=model_version, 
                    temperature=.1)
    return llm
  else:
    raise Exception(f"Model Version {model_version} not supported")
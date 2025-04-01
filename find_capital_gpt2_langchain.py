from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load GPT-2 model using Hugging Face pipeline
hf_pipeline = pipeline(
    "text-generation", 
    model="gpt2",
    max_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2
)

# Wrap it in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Generate text using LangChain
response = llm("What is the capital of India?")

print(response)

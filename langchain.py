from transformers import pipeline

# Load GPT-2 model
llm = pipeline("text-generation", model="gpt2")

# Generate text with constraints
response = llm(
    "What is the capital of India?",
    max_length=50,  # Limit output length
    temperature=0.7,  # Balance randomness
    top_k=50,  # Consider only top 50 token probabilities
    top_p=0.9,  # Use nucleus sampling
    repetition_penalty=1.2  # Penalize repeated tokens
)

print(response[0]["generated_text"])
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







# # ============================================================
# # 🧠 EXPLANATION OF YOUR CODE (TRANSFORMERS PIPELINE)
# # ============================================================

# # ------------------------------------------------------------
# # 🔹 1. WHAT IS NUCLEUS SAMPLING (top_p)?
# # ------------------------------------------------------------

# """
# Nucleus Sampling = Top-p Sampling

# 👉 Idea:
# Instead of picking from ALL possible tokens,
# we only consider a subset of tokens whose cumulative probability ≥ p

# Example:
# Token probabilities:
# A → 0.4
# B → 0.3
# C → 0.2
# D → 0.1

# If top_p = 0.9:
# 👉 Select A + B + C (0.4 + 0.3 + 0.2 = 0.9)
# 👉 Ignore D

# Then randomly pick from A, B, C

# ✅ Advantage:
# - More natural text
# - Avoids rare/unlikely words

# ❌ If too high (top_p=1):
# - behaves like normal sampling

# 🔥 Summary:
# top_p controls "how much probability mass to consider"
# """


# # ------------------------------------------------------------
# # 🔹 2. WHAT IS (response[0]["generated_text"])?
# # ------------------------------------------------------------

# """
# response = llm(...) returns a LIST of dictionaries

# Example:
# response = [
#     {
#         "generated_text": "What is the capital of India? New Delhi..."
#     }
# ]

# 👉 response[0]
# = first output (since list)

# 👉 response[0]["generated_text"]
# = actual generated text string

# 🔥 So:
# response[0]["generated_text"] → final answer text
# """

# # ------------------------------------------------------------
# 3. Other Parameters:-
# # ------------------------------------------------------------
# repetition_penalty:
# - Avoid repeating words
# - >1 → less repetition
# """


# # ------------------------------------------------------------
# # 🔹 4. PIPELINE TYPES (VERY IMPORTANT 🔥)
# # ------------------------------------------------------------

# """
# Transformers pipeline supports MANY tasks:

# 1️⃣ text-generation ⭐
#    → GPT, LLaMA, etc.
#    → Generate text

# 2️⃣ text2text-generation
#    → T5, FLAN
#    → Input → output transformation

# 3️⃣ summarization
#    → Summarize long text

# 4️⃣ translation
#    → Language translation

# 5️⃣ question-answering
#    → Answer based on context

# 6️⃣ fill-mask
#    → BERT-style masked prediction

# 7️⃣ sentiment-analysis
#    → Positive/negative classification

# 8️⃣ zero-shot-classification
#    → classify without training

# 9️⃣ token-classification
#    → NER (name entity recognition)

# 10️⃣ feature-extraction
#    → embeddings

# 11️⃣ image-classification
# 12️⃣ object-detection
# 13️⃣ speech-recognition
# 14️⃣ text-to-speech

# 🔥 Example:
# pipeline("sentiment-analysis")
# pipeline("summarization")
# """


# # ------------------------------------------------------------
# # 🔹 5. OTHER MODELS YOU CAN USE (INSTEAD OF GPT-2)
# # ------------------------------------------------------------

# """
# 🟢 SMALL MODELS:
# - gpt2
# - distilgpt2

# 🟡 MEDIUM:
# - EleutherAI/gpt-neo-1.3B
# - facebook/opt-1.3b

# 🔴 LARGE:
# - meta-llama/Llama-2-7b
# - mistralai/Mistral-7B

# 👉 Note:
# Large models require GPU
# """


# # ------------------------------------------------------------
# # 🔹 6. COMPLETE FLOW (PIPELINE)
# # ------------------------------------------------------------

# """
# Input prompt
# ↓
# Tokenizer (text → tokens)
# ↓
# Model (predict next token probabilities)
# ↓
# Sampling (top_k, top_p, temperature)
# ↓
# Generate tokens
# ↓
# Convert tokens → text
# ↓
# Return list of outputs
# """


# # ------------------------------------------------------------
# # 🎯 INTERVIEW ONE-LINER
# # ------------------------------------------------------------

# """
# "Nucleus sampling selects tokens from the smallest set whose cumulative 
# probability exceeds a threshold p, ensuring a balance between diversity 
# and coherence in generated text."
# """

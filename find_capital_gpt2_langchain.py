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








# ------------------------------------------------------------
# 🔹 1. WHAT IS THIS LINE DOING?
# ------------------------------------------------------------

llm = HuggingFacePipeline(pipeline=hf_pipeline)

"""
👉 Meaning:

- hf_pipeline = HuggingFace Transformers pipeline (your model + tokenizer + task)
- HuggingFacePipeline = LangChain wrapper

👉 Why wrap?

LangChain expects a "standard LLM interface"

So this line:
👉 Converts HuggingFace model → LangChain-compatible LLM

🔥 Without this:
You cannot plug HuggingFace model into:
- Chains
- Agents
- RAG pipelines
"""

# ------------------------------------------------------------
# 🔄 FLOW
# ------------------------------------------------------------

"""
Your Flow:

User Input
↓
LangChain (llm object)
↓
HuggingFacePipeline wrapper
↓
hf_pipeline (actual model)
↓
Model generates output
↓
Returned to LangChain
"""


# ------------------------------------------------------------
# 🔹 2. HOW hf_pipeline IS CREATED
# ------------------------------------------------------------

from transformers import pipeline

hf_pipeline = pipeline(
    "text-generation",   # task
    model="gpt2",        # model
    max_length=50,
    temperature=0.7
)

"""
👉 This is raw HuggingFace pipeline
👉 Works independently (no LangChain)
"""


# ------------------------------------------------------------
# 🔹 3. OTHER PIPELINE TASK OPTIONS (VERY IMPORTANT 🔥)
# ------------------------------------------------------------

"""
🔥 NLP TASKS:

"text-generation"
→ Generate text (GPT, LLaMA)

"text2text-generation"
→ Input → output transformation (T5)

"summarization"
→ Summarize long text

"translation"
→ Translate language

"question-answering"
→ Answer based on context

"fill-mask"
→ Fill missing word (BERT)

"sentiment-analysis"
→ Positive/negative

"zero-shot-classification"
→ Classify without training

"token-classification"
→ NER (names, places)

"feature-extraction"
→ Get embeddings


🔥 MULTIMODAL TASKS:

"image-classification"
"object-detection"
"image-to-text"
"text-to-image"

"automatic-speech-recognition"
"text-to-speech"
"""


# ------------------------------------------------------------
# 🔹 4. PIPELINE PARAMETERS (IMPORTANT)
# ------------------------------------------------------------

pipeline(
    task="text-generation",
    model="gpt2",
    tokenizer="gpt2",  # optional (auto-loaded)
    device=0,          # GPU (0) or CPU (-1)
    batch_size=1
)

"""
👉 device:
- -1 → CPU
- 0 → GPU

👉 tokenizer:
- Converts text → tokens
"""


# ------------------------------------------------------------
# 🔹 5. GENERATION PARAMETERS (INSIDE PIPELINE CALL)
# ------------------------------------------------------------

hf_pipeline(
    "Hello",
    max_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)

"""
👉 These control output style
"""


# ------------------------------------------------------------
# 🔹 6. LANGCHAIN VS RAW PIPELINE
# ------------------------------------------------------------

"""
Raw HuggingFace:
+ Simple
+ Direct use

LangChain Wrapper:
+ Works with chains
+ Works with agents
+ Works with RAG

👉 Use LangChain when:
- building systems

👉 Use raw pipeline when:
- testing models
"""


# ------------------------------------------------------------
# 🔹 7. OTHER LANGCHAIN MODEL WRAPPERS (IMPORTANT 🔥)
# ------------------------------------------------------------

"""
Instead of HuggingFacePipeline, you can use:

1️⃣ OpenAI / ChatOpenAI
   → GPT, DeepSeek, etc.

2️⃣ Ollama
   → local LLMs

3️⃣ AzureChatOpenAI
   → enterprise

4️⃣ Cohere
   → embeddings + LLM

5️⃣ Anthropic (Claude)

👉 HuggingFacePipeline = for local HF models
"""


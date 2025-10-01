from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
# load model from HuggingFace
# def load_model(model_name="context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"):
#     ## load model and tokenizer
#     # Configure quantization for memory efficiency
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.float16,
#     )
#
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
#
#     print("✅ Model and tokenizer loaded")
#     return model, tokenizer

# -----------------------------
# Prompt Template
# -----------------------------

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an experienced assistant specializing in question-answering tasks.
    Utilize the provided context to respond to the question.

    Rules:
      - If the question refers to a **specific table**, note that tables may be identified by either:
        • Roman numerals (I, II, III, IV, …)
        • Arabic numerals (1, 2, 3, 4, …)
      - Normalize references (e.g., "Table II" = "Table 2"). Always check both forms when matching.
      - Only answer using information contained in that table.
      - If the table is not found or the requested information is not in the table, respond with: "I don't know."

      - If the question is about a **formula**:
        • Extract the formula from the context (in LaTeX).
        • Present it in a clean readable way:
            - Use a block math display for clarity: $$ ... $$
            - Then rewrite it inline in plain text (e.g., f_final^t = β·f_adapter^t + (1 - β)·f_original^t).
        • Briefly explain what each symbol means if the context provides that information.
        • If the formula is not found, respond with: "I don't know."

      - If the question is not about a table or a formula, answer using the context as normal.
      - Never provide an answer you are unsure about.
      - Keep answers concise, factual, and easy for non-experts to read.

    CONTEXT:
    {context}

    QUESTION: {question}

    DETAILED RESPONSE:
    """,
    input_variable=["context", "question"]
)

#call api groq
llm = ChatGroq(
api_key=os.environ.get("GROQ_API_KEY"),
model="meta-llama/llama-4-scout-17b-16e-instruct",
temperature=0.3,
max_tokens=1024
)

print("✅ Using Groq LLM")

# Function to ask a question
def ask_question(question, context):
    final_prompt = prompt.invoke({"context": context, "question": question})
    answer = llm.invoke(final_prompt)
    return answer.content


















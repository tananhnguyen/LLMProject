# Model utility functions
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def get_minilm_qa_model(model_name="microsoft/MiniLM-L12-H384-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model
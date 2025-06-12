from fastapi import FastAPI
from pydantic import BaseModel
from finetuning.deploy.prediction import Llama7BInference


app = FastAPI()
model_dir = "models/finetune_model.pt"
tokenizer_dir = "models/tokenizer"
infer = Llama7BInference(model_dir, tokenizer_dir)

class PromptInput(BaseModel):
    context: str
    question: str

@app.post("/generate")
def generate_text(data: PromptInput):
    context = data.context
    question = data.question
    prompt = infer.transform_prompt(context, question)
    result = infer.predict(prompt)
    return {"response": result}

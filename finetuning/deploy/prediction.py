import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Llama7BInference:
    def __init__(self, model_path, tokenizer_path, device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()
        self.device = device

    def predict(self, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device) 

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # <-- Pass it here!
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,     # KV cache
                do_sample=True,   
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()
    
    def transform_prompt(self, context, question):
        user_text = f"Context: {context} Question: {question}"
        return f"<s>[INST] {user_text} [/INST]"
    
model_dir = "models/finetune_model.pt"
tokenizer_dir = "models/tokenizer"
llama_extractor = Llama7BInference(model_dir, tokenizer_dir)

# Usage
if __name__ == "__main__":
    model_dir = "models/finetune_model.pt"
    tokenizer_dir = "models/tokenizer"
    infer = Llama7BInference(model_dir, tokenizer_dir)

    print("==> Positive test case:")
    context = "The capital of the USA is New York. It is a major city known for its influence in finance, culture, and politics. Convex optimization is a subfield of mathematical optimization that studies the problem of minimizing convex functions over convex sets (or, equivalently, maximizing concave functions over convex sets). Many classes of convex optimization problems admit polynomial-time algorithms,[1] whereas mathematical optimization is in general NP-hard.[2][3][4]"
    question = "What is the capital of USA?"
    prompt = infer.transform_prompt(context, question)
    result = infer.predict(prompt)
    print(result)

    print("==> Negative test case:")
    question = "What is the capital of France?"
    prompt = infer.transform_prompt(context, question)
    result = infer.predict(prompt)
    print(result)

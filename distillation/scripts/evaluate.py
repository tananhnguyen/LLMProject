# Evaluation script
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# from utils.metrics import compute_metrics
from transformers import default_data_collator

from utils.data_utils import load_squad_v2
from utils.model_utils import get_minilm_qa_model
from utils.metrics import squad_evaluate
from tqdm import tqdm

def get_predictions(model, tokenizer, dataset):
    import torch
    model.eval()
    predictions = {}
    for example in tqdm(dataset, desc="Predicting"):
        inputs = tokenizer(
            example["question"].strip(),
            example["context"],
            return_tensors="pt",
            truncation="only_second",
            max_length=384,
            stride=128,
            padding="max_length"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]
            start_idx = int(start_logits.argmax())
            end_idx = int(end_logits.argmax())
            input_ids = inputs["input_ids"][0]
            answer = tokenizer.decode(input_ids[start_idx:end_idx+1], skip_special_tokens=True)
            predictions[example["id"]] = answer
    return predictions

def get_references(dataset):
    references = {}
    for example in dataset:
        answers = example["answers"]["text"]
        references[example["id"]] = answers
    return references

def main():
    model_dir = os.path.join("models", "student").replace("\\", "/")
    tokenizer, model = get_minilm_qa_model(model_name=model_dir)
    dataset = load_squad_v2()
    val_set = dataset["validation"]

    predictions = get_predictions(model, tokenizer, val_set)
    references = get_references(val_set)
    metrics = squad_evaluate(predictions, references)
    print("Evaluation Results:", metrics)

if __name__ == "__main__":
    main()
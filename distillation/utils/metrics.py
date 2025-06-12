from datasets import load_metric
import collections
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return ' '.join([w for w in text.split() if w.lower() not in ('a', 'an', 'the')])
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def squad_evaluate(predictions, references):
    """predictions: dict[qid] = pred_answer, references: dict[qid] = gold_answer(s)"""
    total = len(references)
    exact = f1 = 0
    for qid, gold_answers in references.items():
        if qid not in predictions:
            continue
        pred = predictions[qid]
        golds = gold_answers if isinstance(gold_answers, list) else [gold_answers]
        if not golds or golds == [""]:
            # No gold answer: treat as unanswerable, compare to empty string
            exact += int(pred.strip() == "")
            f1 += int(pred.strip() == "")
        else:
            exact += max(compute_exact(g, pred) for g in golds)
            f1 += max(compute_f1(g, pred) for g in golds)
    exact = 100.0 * exact / total
    f1 = 100.0 * f1 / total
    return {"exact_match": exact, "f1": f1}
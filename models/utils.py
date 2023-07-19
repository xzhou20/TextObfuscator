
import torch
from rouge_score import rouge_scorer


def cosine_similarity(x1, x2):
    """
    Input:
        x1 (num1, hidden_size)
        x2 (num2, hidden_size)
    Output:
        similarity (num1, num2)
    
    """
    x1 = (x1 / torch.norm(x1, dim=-1, keepdim=True))
    x2 = (x2 / torch.norm(x2, dim=-1, keepdim=True))
    similarity = torch.mm(x1, x2.T)
    return similarity

def rouge(input_ids, pred_ids, tokenizer):
    batch_real_tokens = [tokenizer.decode(item, skip_special_tokens=True) for item in input_ids]
    batch_pred_tokens = [tokenizer.decode(item, skip_special_tokens=True) for item in pred_ids]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    # scores = scorer.score('The quick brown fox jumps over the lazy dog',
    #                   'The quick brown dog jumps on the log.')
    hit_cnt = 0
    total_cnt = 0
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        rouge_score = scorer.score(real_tokens, pred_tokens)['rougeL'].fmeasure
        hit_cnt += rouge_score
        total_cnt += 1
    return hit_cnt, total_cnt

def token_hit(input_ids, pred_ids, tokenizer, special_tokens):
    batch_real_tokens = [tokenizer.convert_ids_to_tokens(item) for item in input_ids]
    if pred_ids.shape[-1]==1:
        pred_ids = pred_ids.unsqueeze(0)
    batch_pred_tokens = [tokenizer.convert_ids_to_tokens(item) for item in pred_ids]
    hit_cnt = 0
    total_cnt = 0
    special_tokens = ['<s>', '</s>', '<pad>']
    for real_tokens, pred_tokens in zip(batch_real_tokens, batch_pred_tokens):
        real_tokens = {item.replace('Ġ', '').lower() for item in set(real_tokens) if item not in special_tokens}
        pred_tokens = {item.replace('Ġ', '').lower() for item in set(pred_tokens) if item not in special_tokens}
        hit_cnt += len(real_tokens & pred_tokens)
        total_cnt += len(real_tokens)
    return hit_cnt, total_cnt

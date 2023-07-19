from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

task_name = 'sst2'

def simple_tokenizer(text):
    return text.split()

def tokenize_without_subword(text):
    tokenized = tokenizer.tokenize(text)
    subword_ids = tokenizer(*[text], padding=False).word_ids(batch_index=0)
    subword_ids = subword_ids[1:-1]
    words = []
    previous_word_idx = None
    for word, word_idx in  zip(tokenized, subword_ids):
        # if word_idx is None:
        #     pass
        if word_idx != previous_word_idx:
            words.append(word)
        # else:
        #     pass
        previous_word_idx = word_idx
    return ' '.join(words)
    # tokenized = [item if '' in item ]
    

data = load_dataset("glue", task_name)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

label_split_data = {}
if task_name == 'sst2':
    for sentence, label in  zip(data['train']['sentence'], data['train']['label']):
        if label not in label_split_data:
            label_split_data[label] = ''
        # label_split_data[label] = ' '.join([label_split_data[label], sentence])
        text = sentence
        label_split_data[
            label] = ' '.join([label_split_data[label], tokenize_without_subword(text)])

elif task_name == 'mrpc':
    for sentence1, sentence2, label in  zip(data['train']['sentence1'], data['train']['sentence2'], data['train']['label']):
        if label not in label_split_data:
            label_split_data[label] = ''
        text = sentence1 + ' ' + sentence2
        label_split_data[label] = ' '.join([label_split_data[label], tokenize_without_subword(text)])
        # label_split_data[label] = ' '.join([label_split_data[label], sentence1, sentence2])

corpus = []
for label_id, document in label_split_data.items():    
    corpus.append(document)
tr_idf_model  = TfidfVectorizer(tokenizer=simple_tokenizer, lowercase=False)
tf_matrix = tr_idf_model.fit_transform(corpus)
tf_matrix = torch.tensor(tf_matrix.toarray())

idf_matrix = torch.log(torch.tensor(tr_idf_model.idf_))
tf_idf_matrix = tf_matrix * idf_matrix
topk_word_ids = torch.topk(tf_idf_matrix, k=100)[1]
topk_word = tr_idf_model.get_feature_names_out()[topk_word_ids]
# print(f'label {label_id}\n topk idf word \n {topk_word}')
print(topk_word.tolist())
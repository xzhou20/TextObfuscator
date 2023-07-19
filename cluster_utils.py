from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
import torch
import json
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, default_data_collator, DataCollatorWithPadding, DataCollatorForTokenClassification


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def bulid_sentence_dataloader(task_name='sst2', tokenizer=None , max_length=128, batch_size=128):
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of= None)
    if task_name in task_to_keys:
        raw_datasets = load_dataset("glue", task_name)
        sentence1_key, sentence2_key = task_to_keys[task_name]
    else:
        raw_datasets = load_dataset(task_name)
        sentence1_key, sentence2_key = ('text', None)
    padding =   False
    max_length = 512
    
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)
        subword_masks = []
        for  i, sentence in enumerate(result['input_ids']):
            word_ids = result.word_ids(batch_index=i)
            previous_word_idx = None
            subword_mask = []
            for word_idx in word_ids:
                if word_idx is None:
                    subword_mask.append(1)
                elif word_idx != previous_word_idx:
                    subword_mask.append(1)
                else:
                    subword_mask.append(0)
                previous_word_idx = word_idx
            subword_masks.append(subword_mask)
            
        if "label" in examples:
                # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        result['special_tokens_mask'] = subword_masks
        return result

    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"] if 'validation' in processed_datasets else processed_datasets["test"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
    return train_dataloader, eval_dataloader

def bulid_token_dataloader(task_name=None, tokenizer=None, train_file=None, eval_file=None, max_length=128, batch_size=128):
    data_collator = DataCollatorForTokenClassification(
            tokenizer
        )
    
    if task_name == None:
        data_files = {}
        data_files["train"] = train_file
        data_files["validation"] = eval_file
        extension =  train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        raw_datasets = load_dataset(task_name)
    
    # raw_datasets = load_dataset("glue", task_name)
    # sentence1_key, sentence2_key = task_to_keys[task_name]
    text_column_name = 'tokens'
    padding =   False
    max_length = 512
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        
        subword_masks = []
        for  i, sentence in enumerate(tokenized_inputs['input_ids']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            subword_mask = []
            for word_idx in word_ids:
                if word_idx is None:
                    subword_mask.append(1)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    subword_mask.append(1)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    subword_mask.append(0)
                previous_word_idx = word_idx
            subword_masks.append(subword_mask)
        tokenized_inputs['special_tokens_mask'] = subword_masks
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    
    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
    
    return train_dataloader, eval_dataloader

def generate_token_embeddings(model, dataloader, target_layer=3, use_subword=True):
    total_tokens = {}
    token_cnt = {}
    model.eval()
    pro_bar = tqdm(range(len(dataloader)))
    step=0
    
    hidden_size = model.config.hidden_size
    device = model.device
    pro_bar.set_description('generating contextual embedding')
    candicate_embedding = torch.zeros(model.config.vocab_size, hidden_size, device=device)
    token_cnt = torch.zeros(model.config.vocab_size, device=device)
    for batch in dataloader:
        step+=1
        with torch.no_grad():
            batch = {key:value.to(device) for key,value in batch.items()}
            batch['output_hidden_states'] = True
            if 'special_tokens_mask' in batch:
                special_tokens_mask = batch.pop('special_tokens_mask')
            batch.pop('labels')
            outputs = model(**batch)
            input_ids = batch['input_ids']
            target_hidden_states = outputs.hidden_states[target_layer]
            
            attention_mask = batch['attention_mask'] 
            if use_subword:
                valid_ids = (attention_mask!=0) 
            else:
                valid_ids = (attention_mask!=0) & (special_tokens_mask!=0)
            
            input_ids = input_ids[valid_ids]
            target_hidden_states = target_hidden_states[valid_ids]
            
            candicate_embedding.scatter_add_(0, input_ids.unsqueeze(1).repeat(1, 768), target_hidden_states)
            token_cnt.scatter_add_(0, input_ids, torch.ones(input_ids.shape).type_as(token_cnt))
            
        pro_bar.update(1)
    
    valid_token_embeddings_ids = token_cnt.nonzero().squeeze()
    token_embeddings = candicate_embedding[valid_token_embeddings_ids]
    token_embeddings = (token_embeddings/token_cnt[valid_token_embeddings_ids].unsqueeze(1)).cpu()
    sample2vocab = {sample_ids:vocab_ids.item() for sample_ids, vocab_ids in enumerate(valid_token_embeddings_ids)}
    
    return token_embeddings, sample2vocab

def generate_token_prototype(model, dataloader, target_layer=3, use_subword=True):
    total_tokens = {}
    token_cnt = {}
    model.eval()
    pro_bar = tqdm(range(len(dataloader)))
    step=0
    
    hidden_size = model.config.hidden_size
    device = model.device
    pro_bar.set_description('generating contextual embedding')
    candicate_prototype = torch.zeros(len(model.config.id2label), hidden_size, device=device)
    token_cnt = torch.zeros(len(model.config.id2label), device=device)
    for batch in dataloader:
        step+=1
        with torch.no_grad():
            batch = {key:value.to(device) for key,value in batch.items()}
            batch['output_hidden_states'] = True
            if 'special_tokens_mask' in batch:
                special_tokens_mask = batch.pop('special_tokens_mask')
            outputs = model(**batch)
            
            label_ids = batch['labels']
            target_hidden_states = outputs.hidden_states[target_layer]
            
            attention_mask = batch['attention_mask'] 
            valid_ids = (attention_mask!=0) & (label_ids!=-100)
        
            label_ids = label_ids[valid_ids]
            target_hidden_states = target_hidden_states[valid_ids]

            candicate_prototype.scatter_add_(0, label_ids.unsqueeze(1).repeat(1, 768), target_hidden_states)
            token_cnt.scatter_add_(0, label_ids, torch.ones(label_ids.shape).type_as(token_cnt))

        pro_bar.update(1)
    candicate_prototype = (candicate_prototype/token_cnt.unsqueeze(1)).cpu()
    return candicate_prototype

def generate_hierachy_center(cluster_results, contextual_embedding, cluster_num):
    cluster_center = torch.zeros((cluster_num, contextual_embedding.shape[1]))
    token_cnt = torch.zeros(cluster_num)
    
    for sample_ids, cluster_ids in enumerate(cluster_results):
        cluster_center[cluster_ids] += contextual_embedding[sample_ids]
        token_cnt[cluster_ids] += 1
    
    for i in range(cluster_num):
        cluster_center[i] /= token_cnt[i]
    
    return cluster_center
        

import time
def run_cluster(model, dataloader, tokenizer, LR_word=[], target_layer=3, cluster_num=100, cluster_method='kmeans', use_subword=True):
    token_embeddings, sample2vocab = generate_token_embeddings(model, dataloader, target_layer, use_subword=use_subword)
    
    print(f'run {cluster_method} clustering...')
    start_time = time.time()
    if cluster_method == 'kmeans':
        clusters = KMeans(n_clusters=cluster_num, random_state=0).fit(token_embeddings)
        cluster_results = clusters.predict(token_embeddings)
        cluster_center = torch.tensor(clusters.cluster_centers_).to(model.device)
    elif cluster_method == 'hierarchy':
        clusters = AgglomerativeClustering(n_clusters=cluster_num).fit(token_embeddings)
        cluster_results = clusters.labels_
        cluster_center = generate_hierachy_center(cluster_results, token_embeddings, cluster_num)
    elif cluster_method == 'gmm':
        from sklearn.mixture import GaussianMixture
        clusters = GaussianMixture(n_components=cluster_num, random_state=0).fit(token_embeddings)
        cluster_results = clusters.predict(token_embeddings)
        cluster_center = torch.tensor(clusters.means_).to(model.device)
    print(f'run {cluster_method} clustering...done! cost {time.time()-start_time}')
    token2cluster = {}
    for sample_ids, cluster_ids in enumerate(cluster_results):
        token2cluster[sample2vocab[sample_ids]] = int(cluster_ids.item())
    return token2cluster, cluster_center, token_embeddings, sample2vocab

def redivide_cluster(label_related_words, cluster_center, tokenizer, token2cluster, token_embeddings, embeddings_to_vocabulary, cluster_num):
    label_related_words_ids = [tokenizer.convert_tokens_to_ids(item) for item in label_related_words]
    LRWords_cluster = [[token2cluster[token_id] for token_id in label_i] for label_i in label_related_words_ids]
    LRWords_cluster = torch.tensor(LRWords_cluster)
    
    label_num, topk = LRWords_cluster.shape
    vocab_to_embeddings = {int(value):int(key) for key,value in embeddings_to_vocabulary.items()}
    
    for i in range(label_num):
        label_i_topk = LRWords_cluster[i]
        label_i_clusters = set(label_i_topk.tolist())
        for j in range(i+1, label_num):
            label_j_topk = LRWords_cluster[j]
            conflit_mask = (label_i_topk.unsqueeze(1) == label_j_topk.unsqueeze(0))
            conflit_pairs = conflit_mask.nonzero()

            for conflit_item in conflit_pairs:
                token_ids = label_related_words_ids[j][conflit_item[1]]
                token_embedding = token_embeddings[vocab_to_embeddings[token_ids]]
                candicate_cluster = torch.topk(torch.cdist(token_embedding.unsqueeze(0).type_as(cluster_center), cluster_center), k=min(30, cluster_num))[1].squeeze()
                for candicate in candicate_cluster:
                    candicate = candicate.item()
                    if candicate not in label_i_clusters:
                        token2cluster[token_ids] = candicate
                        break
    ori_cluster_center = cluster_center.cpu().detach().clone()
    cluster_center =  [[] for i in range(cluster_num)]
    for token_id, cluster_id in token2cluster.items():
        if token_id == 1:
            continue
        cluster_center[cluster_id].append(token_embeddings[vocab_to_embeddings[token_id]])
    for cluster_id in range(cluster_num):
        cluster_center[cluster_id] = torch.stack(cluster_center[cluster_id]).mean(dim=0)
    cluster_center = torch.stack(cluster_center)
    print(torch.nn.functional.cosine_similarity(ori_cluster_center, cluster_center))
    return token2cluster, cluster_center

import random
def random_cluster(path, save_path, cluster_num=100):
    token2cluster = json.load(open(path, 'r'))
    token2cluster = {int(key):random.randint(0, cluster_num-1) for key,value in token2cluster.items()}
    with open(save_path,"w") as f:
        f.write(json.dumps(token2cluster))
   
def save_results(token2cluster, cluster_center, output_dir, prefix=''):
    unique, counts = np.unique(list(token2cluster.values()), return_counts=True)
    plt.figure(figsize=(25, 10))
    plt.bar(unique, counts)
    plt.title(f'{prefix} result statistics')
    plt.xlabel('cluster center index')
    plt.ylabel('num of tokens')
    plt.savefig(f'{output_dir}/{prefix}_vis.png')
    with open(f'{output_dir}/{prefix}_token2cluster.json',"w") as f:
        f.write(json.dumps(token2cluster))
    if cluster_center != None:
        torch.save(cluster_center.squeeze(), f'{output_dir}/{prefix}_cluster_center.pt')
    
 
def cluster_pipeline(task_name='sst2', model_name='roberta-base', target_layer=3, cluster_method='kmeans', cluster_num=100, base_output_dir='./', use_subword=True):
    sentence_task = ['sst2', 'ag_news']
    token_task = ['conll2003']
    model_name_or_path = model_name
    batch_size=256
    device = 'cuda'
    
    output_dir = f'{base_output_dir}/{task_name}/{model_name_or_path}/{cluster_method}/layer{target_layer}/cluster{cluster_num}'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save_name = f'layer{target_layer}_{cluster_num}'
    
    if task_name in token_task:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    from transformers import RobertaForSequenceClassification
    model = RobertaForSequenceClassification.from_pretrained(model_name_or_path,
                                                               from_tf=bool(".ckpt" in model_name_or_path),
                                                            config=config)
    
    
    if task_name in sentence_task:
        train_dataloader, eval_dataloder = bulid_sentence_dataloader(task_name, tokenizer)
    elif task_name in token_task:
        train_dataloader, eval_dataloder = bulid_token_dataloader(task_name, tokenizer)
        
    model.to(device)
    
    if task_name == 'sst2':
        label_related_words = [['Ġpoorly', 'Ġpointless', 'Ġtires', 'Ġunnecessary', 'Ġbadly', 'Ġunimagin', 'thin', 'Ġlousy', 'Ġcrap', 'Ġjunk', 'Ġirritating', 'Ġplotting', 'Ġpreach', 'Ġbother', 'Ġinsult', 'Ġvap', 'Ġsour', 'Ġnowhere', 'Ġsuffers', 'Ġpit', 'Ġclueless', 'Ġdubbed', 'Ġhole', 'Ġunl', 'Ġexploitation', 'Ġhastily', 'Ġcopy', 'Ġlifeless', 'Ġscreaming', 'falls', 'Ġsleep', 'Ġvague', 'Ġdisguise', 'Ġwooden', 'Ġlackluster', 'poor', 'Ġgarbage', 'Ġthinks', 'Ġinsulting', 'Ġital', 'Ġexcruciating', 'Ġappears', 'Ġthrown', 'Ġloads', 'Ġmistake', 'Ġatro', 'Ġpr', 'Ġcob', 'Ġboat', 'Ġobligatory', 'Ġloose', 'Ġradical', 'Ġdrown', 'Ġsluggish', 'Ġredundant', 'Ġpity', 'Ġconco', 'Ġcostly', 'Ġlower', 'Ġrushed', 'Ġlaz', 'Ġquarter', 'Ġtransparent', 'Ġnonsense', 'Ġdisposable', 'Ġmeaningless', 'lost', 'Ġguess', 'Ġheads', 'Ġleak', 'Ġpool', 'Ġidiots', 'Ġexact', 'Ġwear', 'Ġbusy', 'Ġemptiness', 'Ġmil', 'Ġobscure', 'Ġdistant', 'control', 'Ġdisappointment', 'Ġturf', 'Ġsadd', 'Ġsplit', 'Ġstiff', 'ug', 'Ġpub', 'Ġchees', 'worst', 'skip', 'Ġetc', 'Ġruined', 'Ġnonexistent', 'Ġlimitations', 'Ġpopulated', 'Ġhist', 'Ġcage', 'Ġwarriors', 'Ġawkwardly', 'Ġarbitrary'], ['beaut', 'ple', 'Ġplayful', 'Ġbreathtaking', 'eng', 'Ġdetailed', 'Ġtender', 'Ġrive', 'Ġwonderfully', 'Ġdazzling', 'cle', 'Ġrewarding', 'Ġhearts', 'solid', 'Ġunpredictable', 'Ġhopeful', 'Ġupl', 'Ġtears', 'Ġelegant', 'Ġoutstanding', 'powerful', 'Ġvibrant', 'Ġpleasing', 'Ġrelief', 'Ġstartling', 'Ġvividly', 'Ġenjoyed', 'Ġsympathetic', 'Ġwond', 'Ġlocal', 'Ġahead', 'Ġstirring', 'Ġpotent', 'Ġguessing', 'Ġfluid', 'Ġflow', 'Ġhonesty', 'Ġirresistible', 'Ġsublime', 'Ġpolished', 'Ġopenness', 'Ġtechnology', 'Ġproud', 'Ġconstant', 'Ġbehold', 'scale', 'Ġinviting', 'Ġdelivering', 'Ġdawn', 'bring', 'touch', 'Ġlovers', 'Ġimmensely', 'Ġwhims', 'Ġgy', 'Ġsouls', 'Ġdelicious', 'Ġstrikes', 'Ġrapid', 'Ġembraces', 'Ġunderstands', 'Ġballot', 'Ġenduring', 'Ġencouraging', 'Ġrecovery', 'Ġpopular', 'Ġuncommon', 'Ġmiracle', 'Ġgently', 'Ġsly', 'Ġsensitivity', 'Ġforth', 'Ġsisters', 'Ġcompanion', 'Ġdraws', 'Ġwonders', 'Ġprofile', 'wise', 'Ġelegance', 'Ġcontribution', 'Ġexpressive', 'Ġstatus', 'Ġflashes', 'Ġrecommendation', 'Ġformat', 'Ġchord', 'Ġhay', 'Ġenormously', 'nice', 'Ġgrowth', 'Ġgorge', 'Ġsignificance', 'Ġliberating', 'Ġfinely', 'Ġcheek', 'color', 'Ġconfidence', 'Ġclinic', 'Ġsway', 'Ġties']]
    elif task_name == 'ag_news':
        label_related_words = [['stocks', 'aspx', '=/', 'http', '=', 'www', '="', '://', 'full', 'ĠWireless', 'Ġquarterly', 'ĠAirways', 'Ġsoftware', 'Ġconsumer', 'ĠMicrosoft', 'ĠSecurities', 'Ġinflation', 'ĠStores', 'ĠIBM', 'ĠMonetary', 'Ġcarrier', 'Ġmortgages', 'OR', 'ĠMortgage', 'ĠOv', 'Ġlender', 'Ġyen', 'ĠQuarterly', 'ĠTreasury', 'Nik', 'largest', 'ĠGrowth', 'ĠOPEC', 'Ġconsumers', 'ĠLower', 'ĠSprint', 'ĠIntel', 'NYSE', 'Ġestimates', 'Ġwireless', 'ĠSystems', 'ĠIPO', 'ĠMarsh', 'Trump', 'ĠRouse', 'Ġexports', 'Ġattendants', 'Google', 'EU', 'Ġretailers', 'ĠDollar', 'ĠMcL', 'Oracle', 'ĠBrands', 'Ġbroker', 'Ġsecurities', 'Ġhedge', 'Ġinventory', 'ĠCisco', 'ĠInvestors', 'ĠComputer', 'ĠGoldman', 'ĠAnimation', 'ĠSears', 'ĠGlobal', 'KK', 'Delta', 'OB', 'Microsoft', 'ĠFoods', 'chip', 'PK', 'ĠIMF', 'Ġasbestos', 'ĠMutual', 'ĠSachs', 'ĠEconomic', 'Ġinvent', 'ĠMond', 'ĠOrganization', 'Ġarthritis', 'ĠYahoo', 'Quote', 'Ġairlines', 'Ġlending', 'ĠAccounting', 'ny', 'HP', 'UAL', 'Ġacquisitions', 'ĠAbbey', 'Ġgrocer', 'Econom', 'Ġtech', 'Ġequity', 'ĠConsumer', 'ĠEconomy', 'ĠDepot', 'Ġloans', 'Ġgasoline'], ['ĠMicrosoft', 'Ġsoftware', '=', 'ĠiPod', 'ĠWindows', 'ĠYahoo', 'Microsoft', 'ĠMozilla', 'ĠLinux', 'ĠWireless', '="', 'ĠIBM', 'Ġq', 'info', 'sym', 'ĠServer', 'ĠCisco', 'http', 'ĠNASA', 'ĠIntel', 'NASA', 'ĠFirefox', 'ĠAMD', 'Ġprocessor', 'Ġwireless', '\\\\', 'www', 'ĠDesktop', 'ĠComputer', 'HP', 'ĠSPACE', 'Ġscientists', 'ĠHP', 'ĠIE', 'ĠHubble', 'Ġcomputing', 'uk', 'ĠXP', 'Ġdesktop', 'Ġworm', 'ĠIP', 'ĠMicro', 'ĠSoftware', 'ĠTelescope', 'ĠSolar', 'Ġbrowser', 'Ġserver', 'Google', 'Ġbeta', 'ĠSystems', 'ĠeBay', 'ĠDS', 'ĠMS', 'ĠGenesis', 'ĠXbox', 'Ġstorage', 'ĠWi', 'Ġdevices', 'Intel', 'Quote', 'source', 'Ġspam', '://', 'Ġhackers', 'ĠMessenger', 'ĠiTunes', 'Ġservers', 'ĠSans', 'ĠPCs', 'hel', 'ver', '="#', '6666', 'arial', 'Ġspacecraft', 'Fi', 'Ġapplication', 'ĠBlackBerry', 'ĠGmail', 'Ġtools', 'peer', 'ĠSkype', 'ĠAdobe', 'Ġbroadband', 'Ġconsumer', 'Ġdownload', 'ĠExplorer', 'ĠStorage', 'Ġenterprise', 'ĠNokia', 'Ġconsumers', 'ĠSaturn', 'ĠScientists', 'Linux', 'ĠJava', 'ĠLCD', 'Info', 'Ġspecies', 'ĠHalo', 'ĠUnix'], ['Ġquarterback', 'Sports', 'Ġtouchdown', 'ĠPacers', 'Ġinning', 'ĠPrix', 'Ġstriker', 'Ġtouchdowns', 'ĠEagles', 'Ġpreseason', 'Ġcricket', 'ĠNotre', 'Ġmedal', 'Ġpitcher', 'ĠKnicks', 'ĠChampions', 'Ġcoaching', 'ĠCavaliers', 'ĠTrophy', 'ĠNASCAR', 'ĠBroncos', 'Ġhomer', 'ĠBraves', 'ĠPhillies', 'Ġlinebacker', 'ĠChargers', 'ĠStadium', 'Ġoutfielder', 'ĠMVP', 'Ġunbeaten', 'ĠRedskins', 'Ġrookie', 'ĠRavens', 'NBA', 'ĠSteelers', 'ĠBlackburn', 'ĠFalcons', 'ĠColts', 'ĠRyder', 'Ġchampionship', 'ĠKobe', 'NL', 'ĠPistons', 'ĠBayern', 'ĠValencia', 'ĠNuggets', 'Ġcaptain', 'ĠTimberwolves', 'ĠMariners', 'finals', 'ĠAstros', 'Ġsemifinals', 'Ġqualifier', 'ĠUEFA', 'ĠFul', 'ĠWenger', 'ĠSerie', 'ĠCowboys', 'ĠMasters', 'ĠSPORTS', 'ĠReds', 'ĠChampionships', 'ĠClippers', 'ĠMarlins', 'ĠAdrian', 'ĠChelsea', 'Neal', 'ĠNCAA', 'ĠLakers', 'ĠTrafford', 'ĠNHL', 'ĠVikings', 'Ġbullpen', 'ĠBlazers', 'ĠSeahawks', 'Ġquarterbacks', 'ĠFIFA', 'ĠPatriots', 'ĠGators', 'ĠDame', 'Ġinnings', 'ĠCeltics', 'ĠWizards', 'ĠAthletics', 'ĠTournament', 'ĠEverton', 'ĠKl', 'ĠIndies', 'ĠOwen', 'ĠMets', 'ĠMut', 'Ġundefeated', 'ĠCubs', 'Ġbaseman', 'ĠBundesliga', 'ĠAngels', 'ĠJuventus', 'yard', 'ĠMLS', 'ĠRicky'], ['Ġmilitants', 'ĠAfghan', 'ĠHamas', 'Ġnuclear', 'ĠGaza', 'Ġcleric', 'Ġhostages', 'ĠAra', 'ĠAriel', 'ĠKabul', 'Qaida', 'ĠSunni', 'ĠMosul', 'ĠSaddam', 'Ġmosque', 'ĠIraqis', 'ĠMush', 'ĠShiite', 'Ġenrichment', 'Ġdisarm', 'ĠAbbas', 'ĠTaliban', 'ĠPalestinians', 'ĠSamar', 'Ġkidn', 'Ġbomber', 'ĠSharon', 'Palest', 'Ġsuicide', 'ĠKashmir', 'ĠRebels', 'ĠSyrian', 'Qaeda', 'ĠNaj', 'ĠNepal', 'ĠSinai', 'Ġinsurgent', 'Ġpolicemen', 'ĠRw', 'ĠAbu', 'Ġlandsl', 'ĠMyanmar', 'ĠMahmoud', 'ite', 'Ġconvoy', 'Ġbombing', 'Ġinsurgents', 'ĠHaiti', 'ĠPutin', 'Ġmilitias', 'ĠThatcher', 'Ġwounding', 'ĠMb', 'Ġinsurgency', 'ĠGuantanamo', 'ĠImam', 'ĠLebanese', 'Ġhumanitarian', 'Ġgenocide', 'ĠAfghans', 'ĠYugoslav', 'ĠIslam', 'ĠJordanian', 'ĠKilled', 'EU', 'Israeli', 'Sad', 'ĠHussein', 'ĠFein', 'ĠArist', 'ĠKurdish', 'Ġassassinated', 'Ġsurvivors', 'ĠSyria', 'ĠViktor', 'ĠNep', 'ĠJazeera', 'ĠSinn', 'Ġparamilitary', 'ĠPalestine', 'ĠLaden', 'Ġrepublic', 'Ġgrenade', 'ĠAtomic', 'Gaza', 'ĠDamascus', 'ĠNuclear', 'ĠHaitian', 'Ġkidnapping', 'ĠColin', 'ĠStraw', 'Syria', 'Ġembargo', 'ĠIslamist', 'ĠJewish', 'Ġdetainees', 'ĠEgyptian', 'Ġasylum', 'Ġprisoner', 'ĠLebanon']]
    
    token2cluster, cluster_center, token_embeddings, sample2vocab = run_cluster(model, train_dataloader, tokenizer, label_related_words, target_layer, cluster_num, cluster_method=cluster_method, use_subword=use_subword)
    
    if use_subword:
        prefix='wsubword'
    else:
        prefix='nosubword'
    
    save_results(token2cluster=token2cluster, cluster_center=cluster_center, output_dir=output_dir, prefix=prefix)

    token2cluster, cluster_center = redivide_cluster(label_related_words=label_related_words, cluster_center=cluster_center, tokenizer=tokenizer, token2cluster=token2cluster, token_embeddings=token_embeddings, embeddings_to_vocabulary=sample2vocab, cluster_num=cluster_num)
    
    prefix = prefix + '_tfidf'
    save_results(token2cluster=token2cluster, cluster_center=cluster_center, output_dir=output_dir, prefix=prefix)
    
if __name__ == '__main__':
    task_name = 'sst2'
    model_name='roberta-base'
    target_layer = 3
    # kmeans gmm hierarchy
    cluster_method = 'kmeans'
    cluster_num = 100
    use_subword = True
    
    cluster_pipeline(task_name=task_name, model_name=model_name, target_layer=target_layer, cluster_method=cluster_method, cluster_num=cluster_num, use_subword=use_subword)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModel

class InversionMLP(nn.Module):
    def __init__(self, config):
        super(InversionMLP,self).__init__()

        self.vocab_size = config.vocab_size
        self.input_size = config.hidden_size
        hidden_size=2048
        
        self.model = nn.Sequential(nn.Linear(self.input_size, hidden_size), 
                                nn.ReLU(), 
                                nn.Linear(hidden_size, self.vocab_size))

        self.loss = torch.nn.CrossEntropyLoss()


    def forward(self, x, labels=None, attention_mask=None, token_type_ids=None):
        if attention_mask!=None:
            x *= attention_mask[:,:,None].repeat(1,1,x.shape[-1])
        logits = self.model(x)

        loss = None
        if labels is not None:
            bsz, seq_len, hid_dim = x.shape
            device = x.device
            active_loss  = attention_mask.view(-1) == 1
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss.ignore_index).type_as(labels)
                )
            active_logits = logits.view(-1, self.vocab_size)
            loss = self.loss(active_logits, active_labels)
        return logits, loss

    def predict(self, x, labels=None, attention_mask=None, token_type_ids=None):
        if attention_mask!=None:
            x *= attention_mask[:,:,None].repeat(1,1,x.shape[-1])
            # x *= attention_mask[:,None, :,None].repeat(1,2,1,x.shape[-1])
        logits = self.model(x)
        # logits = self.top_classifier(logits)
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred

class InversionTransformer(nn.Module):
    def __init__(self, config):
        super(InversionTransformer,self).__init__()

        self.vocab_size = config.vocab_size
        self.input_size = config.hidden_size

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=8)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Linear(self.input_size, self.vocab_size)

        self.loss = torch.nn.CrossEntropyLoss()


    def forward(self, x, labels=None, attention_mask=None, token_type_ids=None):
        # if attention_mask!=None:
        #     x *= attention_mask[:,:,None].repeat(1,1,x.shape[-1])
        # attention_mask = (-attention_mask+1).bool()
        logits = self.model(x.permute(1,0,2), src_key_padding_mask=attention_mask.bool()).permute(1,0,2)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            bsz, seq_len, hid_dim = x.shape
            device = x.device
            active_loss  = attention_mask.view(-1) == 1
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss.ignore_index).type_as(labels)
                )
            active_logits = logits.view(-1, self.vocab_size)
            loss = self.loss(active_logits, active_labels)
        return logits, loss

    def predict(self, x, labels=None, attention_mask=None, token_type_ids=None):
        logits = self.model(x.permute(1,0,2), src_key_padding_mask=attention_mask.bool()).permute(1,0,2)
        logits = self.classifier(x)
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred  
    
class InversionPLM(nn.Module):
    def __init__(self, config, model_name_or_path='roberta-base'):
        super(InversionPLM, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, label, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        return outputs.logits, outputs.loss

    def predict(self, x, label=None, attention_mask=None):
        outputs = self.model(inputs_embeds=x, labels=label, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(F.softmax(logits,dim=-1), dim=2)
        return logits, pred

class InversionPLMMLC(nn.Module):
    def __init__(self, config, model_name_or_path='roberta-base'):
        super(InversionPLMMLC,self).__init__()
        self.vocab_size = config.vocab_size
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

        self.loss = torch.nn.BCELoss()
        self.sigmod = torch.nn.Sigmoid()

    def forward(self, x, labels=None, attention_mask=None, token_type_ids=None):
        bsz, seq_len, hid_dim = x.shape
        device = x.device
    
        logits = self.model(inputs_embeds=x, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        
        logits = logits * attention_mask.unsqueeze(-1)
        logits = logits.sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(-1)
        
        logits = self.sigmod(logits) # (bsz, dim)

        loss = None
        if labels is not None:
            labels = torch.zeros(bsz, self.vocab_size).to(device).scatter_(1, labels, 1.)
            labels[:,0:3] = 0
            loss = self.loss(logits, labels)
        return logits, loss

    def predict(self, x, labels=None, attention_mask=None, token_type_ids=None):
        logits = self.model(inputs_embeds=x, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        
        logits = logits * attention_mask.unsqueeze(-1)
        logits = logits.sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(-1)
        logits = self.sigmod(logits)
        
        threshold = 0.5
        pred = (logits > threshold).long()
        return logits, pred

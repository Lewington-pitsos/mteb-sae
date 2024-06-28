from mteb import Encoder
from sae_lens import SAE
from transformer_lens import HookedTransformer
import torch.nn as nn
import transformers

class SAEBaseModel(nn.Module):
    def __init__(self, 
                 transformer_name: str, 
                 sae_release: str,
                 sae_id:str,
                 device, 
                 freeze_sae: bool = True
        ):
        super(SAEBaseModel, self).__init__()

        sae, _, _ = SAE.from_pretrained(release = sae_release, sae_id = sae_id, device = device)

        self.sae = sae

        if freeze_sae:
            for param in self.sae.parameters():
                param.requires_grad = False

        self.device = device

        self.model = HookedTransformer.from_pretrained(transformer_name, device=device)
        self.hook_layer = sae.cfg.hook_layer
        self.hook_name = sae.cfg.hook_name

        for param in self.model.parameters():
            param.requires_grad = False

class SAEFeaturesModel(SAEBaseModel):
    def forward(self, input_ids, attention_mask):
        _, cache = self.model.run_with_cache(input_ids, attention_mask=attention_mask, prepend_bos=True)

        hidden_states = cache[self.hook_name]
        features = self.sae.encode(hidden_states)
        return features

def masked_avg(embedding_matrix, attention_mask):
    attention_mask_expanded = attention_mask.unsqueeze(-1)
    
    sum_embedding = (embedding_matrix * attention_mask_expanded).sum(dim=1)
    non_masked_count = attention_mask.sum(dim=1, keepdim=True)
    
    non_masked_count = non_masked_count.clamp(min=1)
    
    average_embedding = sum_embedding / non_masked_count

    return average_embedding


class SAEEncoder(SAEBaseModel):
    def __init__(self, transformer_name, *args, **kwargs):
        super(SAEEncoder, self).__init__(transformer_name, *args, **kwargs)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, sentences, **kwargs):
        tokens = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

        input_ids, attention_mask = tokens['input_ids'].to(self.device), tokens['attention_mask'].to(self.device)

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        _, cache = self.model.run_with_cache(input_ids, attention_mask=attention_mask, prepend_bos=True)

        hidden_states = cache[self.hook_name]


        features = self.sae.encode(hidden_states)
        
        return masked_avg(features, attention_mask)
import torch
import torch.nn as nn
from transformers import BertLMHeadModel


class PromptEncoder(nn.Module):
    def __init__(self, num_prompt_tokens, hidden_size, word_embedding):
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_embeddings = nn.Embedding(num_prompt_tokens, hidden_size)
        
        init_prompt_value = word_embedding[:num_prompt_tokens].detach().clone()
        self.prompt_embeddings.weight = nn.Parameter(init_prompt_value)
        
        
    def forward(self, prompts):
        return self.prompt_embeddings(prompts)
        
        
        

class PromptTuningBERT(nn.Module):
    def __init__(self, bert_model, num_prompt_tokens=10):
        super().__init__()
        self.bert = BertLMHeadModel.from_pretrained(bert_model)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_encoder = PromptEncoder(num_prompt_tokens, self.bert.config.hidden_size, self.bert.embeddings.word_embeddings.weight)
        
        
    def save_learned_prompts(self, path):
        torch.save(self.prompt_encoder.prompt_embeddings.weight.data, path)
        
    def load_learned_prompts(self, path):
        self.prompt_encoder.prompt_embeddings.weight.data = torch.load(path)


    def forward(self, input_ids, attention_mask):
        prompt_tokens = torch.arange(self.num_prompt_tokens).unsqueeze(0).to(input_ids.device)
        prompt_tokens = prompt_tokens.repeat(input_ids.size(0), 1)
        
        prompt_embeddings = self.prompt_encoder(prompt_tokens)
        input_embeddings = self.bert.wte(input_ids)
        
        input_embeddings = torch.cat(
            [
                prompt_embeddings,
                input_embeddings
            ],
            dim=1
        )
        
        attention_mask = torch.cat(
            [
                torch.ones_like(prompt_tokens).to(input_ids.device),
                attention_mask
            ],
            dim=1
        )

        labels = torch.cat(
            [
                torch.ones_like(prompt_tokens).to(input_ids.device) * -100,
                input_ids
            ],
            dim=1
        )
        
        return self.bert(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )



        
'''
A custom class for LoRA fine-tuning:We take the original model as input and 
add low-rank matrices for fine tuning.

'''
from contextlib import nullcontext
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_llama import LlamaPreTrainedModel, LlamaConfig
from rope import apply_rotary_emb
from classifier import LlamaEmbeddingClassifier
from utils import *

class LoRA_layer(nn.Module):
    def __init__(self,llama_layer,rank,alpha,p = 0):
        super(LoRA_layer, self).__init__()
        self.base_layer = llama_layer
        self.rank = rank
        self.lora_alpha = alpha
        self.p =p
        self.dropout = nn.Dropout(self.p)
        
        self.lora_A = nn.Parameter(torch.zeros(rank, self.base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.base_layer.out_features, rank))
        self.scaling = self.lora_alpha / self.rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        
    def forward(self,x):
        out = self.base_layer(x)
        #Add dropout to reduce overfitting
        if self.p:
            x = self.dropout(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T)*self.scaling
        return out+lora_out
    
   
class addLoraWrapper(nn.Module):
    def __init__(self,model,target_modules,rank,alpha=1,p=0):
        super(addLoraWrapper,self).__init__()
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.p = p
        
        #Replaces the target modules with a LoRA layer
        for name,_ in self.model.named_modules():
            if any(name.endswith(target_name) for target_name in self.target_modules):
                parent = self.model.get_submodule('.'.join(name.split('.')[:-1]))
                child_name = name.split('.')[-1]
                child = self.model.get_submodule(name)
                setattr(parent,child_name,LoRA_layer(child,self.rank,self.alpha,self.p))
       
                    
    def forward(self,x):
        return self.model(x)
                    
                
        
        

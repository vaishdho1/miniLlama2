---
## llama.py
---
### RMSNorm
1. This class computes the Root Mean Square normalization

### Attention
1. This class contains the implementation of multihead cross attention and is derived from the nn.Module class.
2. The functions present in the class are as follows:
    a. compute_query_key_value_scores : The function computes the attention scores and the scaled value vectors.
    b. forward : The forward function in nn.Module is overwritten wiht attention computation followed by dropout. 

### FeedForward
1. This class contains a feed forward layer consisting of a combination of linear layers, SwiGLU activation and dropout

### LlamaLayer
1. This class contains the implementation of one transformer layer
  a. layer normalization of the input (via Root Mean Square layer normalization)
  b. self-attention on the layer-normalized input
  c. a residual connection 
  d. layer normalization on the output of the self-attention
  e. a feed-forward network on the layer-normalized output of the self-attention
  f. a residual connection from the unnormalized self-attention output added to the output of the feed-forward network

### Llama
1. This class contains the implementation of the whole model. This is derived from LlamaPreTrainedModel
2. It contains intial configurations, layers and one complete forward pass through the model giving both the output before tranforming into
   vocabulary size and after transforming into it.
### generate
1. This function generated tokens given a prefix using temperature scaling and sampling from the scaled probability distribution.
2. The function operates in inference mode.

### load_pretrained
1. This function loads the state of the current model from the checkpoint given.

---
## classifier.py
---

### LlamaZeroShotClassifier
1. This class compute the completion probability of each label string using the output from the model.

### LlamaEmbeddingClassifier
1. This class adds an extra classifier head and computes the log-softmax over all the classes.

---
## optimizer.py
---

### AdamW optimizer
1. This class implements the AdamW optimizer from [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) with slight modifications.

---
## rope.py
---
1. The class computes rotary positional embeddings as in RoFormer: Enhanced Transformer with Rotary Position Embedding(https://arxiv.org/abs/2104.09864).

---
## LoRA.py
---

### LoRA_layer
1. This implements one layer of low rank adaptation for fine-tuning.

### addLoraWrapper
1. This converts the target modules specified into LoRA layers.

---
## tokenizer.py
---
1. Contains the implementation of tokenizer similar to the one used in [Llama2 paper](https://arxiv.org/pdf/2307.09288).

---
## run_llama.py
---
1. The file contains
   a. dataset creation
   b.  model evaluation
   c. training the model
   d. generating sentences
   e. main function to picks the args and implement the necessary function.
   

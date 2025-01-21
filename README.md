# miniLlama2

The base structure of the project is picked from Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2024/) assignment.
Pretrained weights are loaded for the language model from stories42M.pt; an 8-layer, 42M parameter language model pretrained on the TinyStories dataset.
The project implements a small version of Llama2 model and performs sentence classification on ``sst`` dataset and ``cfimdb``.

## Key features Implementd

1. Implemented Multiheaded cross attention using RoPE relative positional embeddings.
2. Implemented a Tranformer block.
3. Implemented rotary embeddings.
4. Implemented classification head for sentence classification.
5. Added implementations of Lora Layer and Lora Wrapper.
6. Implemented AdamW optimizer.

## Features tested

1. Generated text completion starting with a prefix.
2. Performed zero-shot prompt based sentiment analysis on the 2 datasets.
3. Performed fine-tuning of the model along with a classification head.
4. Performed parameter efficient fine-tuning using self implementation of LoRa.


## Running the code

1.Run `setup.sh` for intitial configuration.
2.The various commands for running each feature are present in `commands.txt`.

## Results and Outputs

1. The detailed explaination of different result files are present in `results.txt`
2. The best hyperparameters are shown in the table below.

| Dataset | Method       | Epochs | Batch Size | Learning Rate | Alpha | Rank |
|---------|--------------|--------|------------|---------------|-------|------|
| sst     | Fine-tuning  | 5      | 80         | 2e-5          | -     | -    |
| sst     | LoRA         | 5      | 80         | 2e-3          | 1     | 4    |
| cfimdb  | Fine-tuning  | 5      | 10         | 2e-5          | -     | -    |
| cfimdb  | LoRA         | 5      | 10         | 2e-3          | 1     | 4    |


   
4. The table below shows the best sentiment classification accuracies for both the datasets.

| Dataset |       Fine-tuning               |                 LoRA             |
|---------|-------------|---------|---------|--------------|---------|---------|
|         | Train       | Dev     | Test    | Train        | Dev     | Test    |
|---------|-------------|---------|---------|--------------|---------|---------|
| sst     | 0.77        | 0.41    | 0.41    | 0.45         | 0.43    | 0.41    |
| cfimdb  | 0.85        | 0.83    | 0.45    | 0.879        | 0.869   | 0.52    |


### Acknowledgement
This code is based on llama2.c by Andrej Karpathy. Parts of the code are also from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

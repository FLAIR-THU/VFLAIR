# Datasets

 ![Datasets](figures/llm_datasets.png)

In VFLAIR_LLM, we defaultly provide several dataset that are oftenly used for LLM benchmark. As the origianl dataset is not provided along with the codebase, all the dataset provided and guidance on how to achieve these datasets are listed below.

## Default Dataset Usage

  Defaultly, datasets used are stored under directory `../../share_dataset/`.

    1. Sequence Classification/Regression Datasets
       1. GLUE benchmark
          - By default, the data are stored under directory `../../share_dataset/DatasetName/`
          - All 9 datasets in the GLUE benchmark is downloaded from [GLUEDatasets](https://gluebenchmark.com/tasks/) 
       2. Yelp
          - By default, the data are stored under directory `../../share_dataset/Yelp/`
          - Data is downloaded from [YelpDataset](https://huggingface.co/datasets/Yelp/yelp_review_full)
          - Note that you should unzip the data files into 'train.csv' and 'test.csv' before using it.
  2. Text-span based Question Answering Datasets
       1. SQuAD(1.1version)
          - By default, the data are stored under directory `../../share_dataset/SQuAD/`
          - Data is downloaded from [SQuADDataset](https://huggingface.co/datasets/rajpurkar/squad) and arranged into `../../share_dataset/SQuAD/data/dev-v1.1.json` and `../../share_dataset/SQuAD/data/train-v1.1.json`
  3. Generation Datasets
       1. Lambada
          - By default, the data are stored under directory `../../share_dataset/Lambada/`. 
          - Loaded with huggingface.dataset module, using function load_dataset().
          - Data can be downloaded from [LambadaDataset](https://huggingface.co/datasets/cimec/lambada). You can also use other Lambada data sources that support the huggingface.dataset module.
       2. Alpaca
          - Data is downloaded from [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) and arranged into  `../../share_dataset/Alpaca/alpaca_data.json`. 
       3. CodeAlpaca
          - Data is downloaded from [codealpaca](https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K) and arranged into  `../../share_dataset/CodeAlpaca-20k/code_alpaca_20k.json`. 
       4. MATH
          - Data is downloaded from [hendrycks/math: The MATH Dataset](https://github.com/hendrycks/math/)and arranged into  `../../share_dataset/MATH/`. 
       5. GMS8K
          - Data is downloaded from [GMS8KDataset](https://github.com/openai/grade-school-math) and arranged into  `../../share_dataset/GMS8K/`. 

Detailed data loading and processing can be found in `src/load/LoadDataset.py`. You can alter the code in function [load_dataset_per_party_llm] to suit your need.



## Use Your Own Dataset

In VFLAIR-LLM , we're also open for users to implement their own dataset or change default dataset configurations. Main dataset processing is located in `src/load/LoadDataset.py`, where input prompt and configuration is implemented. Tokenization and padding is impelmented in `src/dataset/party_dataset.py`.

You can add your own dataset processing procedure in function [load_dataset_per_party_llm] and class [PassiveDataset_LLM] [LambadaDataset_LLM]....
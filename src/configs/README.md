# How to write a configuration file

## Parameters in a Configuration File

#### 1. Basic VFL Parameters

> Supported Datasets:  All default datasets

```json
"epochs": 30,
"lr": 0.01,
"k": 2,
"batch_size": 1024,
"iteration_per_aggregation": 1,
"dataset":{
"dataset_name": "mnist",
"num_classes": 10
},
"model_list":{
"0": {
"type": "MLP2",
"input_dim": 392,
"output_dim": 10,
"path": "random_14*28_10"
},
"1": {
"type": "MLP2",
"input_dim": 392,
"output_dim": 10,
"path": "random_14*28_10"
},
"apply_trainable_layer": 1,
"global_model": "ClassificationModelHostTrainableHead"
}
```

- "epochs": number of iterations for each experiment
- "lr": main task learning rate
- "k": number of parties
  - While changing "k", model_list shall also be altered.
- "batch_size": #samples used for each batch
- "iteration_per_aggregation": for [FedBCD](https://ieeexplore.ieee.org/abstract/document/9855231/), to increase VFL efficiency. When the value is set to `1`, then FedBCD is not applied, while for value `>1`, FedBCD is applied.

- "dataset": the dataset for experiments
- "dataset_name": name of the dataset, ["mnist", "cifar10", "cifar100", "breast-cancer", "diabetes" ] are supported with default dataset partition, you can add your own dataset if needed by modifying `/src/load/LoadDataset.py` file. 
- "num_classes": number of classes for experiments
- "model_list": specify all the models that are used in the experiment, should be a dictionary with all the party's index as keys
  - "i": for each party with index $i$
  - "type": local model type
  - "input_dim", "output_dim": necessary information to construct a local model
  - "path": specify trained model state dictionary if you want to load pre-trained models

- "apply_trainable_layer": whether the global model is trainable
  - in aggVFL, this parameter shall be '0'; while in splitVFL, it shall be '1'
- "global_model": global model
  - in aggVFL, this parameter shall be name of a global model with no trainable layer, e.g 'ClassificationModelHostHead'
  - in splitVFL, this parameter shall be name of a global model with a trainable layer, e.g 'ClassificationModelHostTrainableHead'
- "attack": attack that is to be evaluated
  - "name": name for the attack, supported attacks list is shown in the bellow
    - TODO: list attack names
  - "parameters": parameters for the specified attack

- "defense": defense that is to be evaluated
  - "name": name for the defense, supported defenses list is shown in the bellow
    - TODO: list attack names
  - "parameters": parameters for the specified defense

#### 2. Attack & Defense

Your can load your own patameters for different attacks like the following example:

##### 2.1 Attack

- BatchLabelReconstruction([Defending batch-level label inference and replacement attacks in vertical federated learning](https://ieeexplore.ieee.org/abstract/document/9833321/))
  - "party": list of party that launch this attack
  - "lr": learning rate for reconstruction model
  - "epochs": iterations for reconstruction model training

```json
"attack_list": {
        "0":{
            "name": "BatchLabelReconstruction",
            "parameters": {
                "party": [0],
                "lr": 0.05,
                "epochs": 10000
            }
        }
      }
```

- ReplacementBackdoor([Defending batch-level label inference and replacement attacks in vertical federated learning](https://ieeexplore.ieee.org/abstract/document/9833321/))
  - "party": list of parties that cooperate with each other to launch this attack

```json
"attack_list": {
        "0": {
            "name": "ReplacementBackdoor",
            "parameters": {
                "party": [0]
            }
        }
      }
```

##### 2.2 Defense

- LaplaceDP ([Differential privacy: A survey of results](https://www.google.com.au/books/edition/Theory_and_Applications_of_Models_of_Com/JHFqCQAAQBAJ?hl=en&gbpv=1&pg=PA1&printsec=frontcover))
  - This is a defense for active party 
  - "dp_strength": laplace noise strength 0.001

```json
"defense": {
        "name": "GaussianDP",
        "parameters": {
            "dp_strength": 0.0001
        }
    }
```

- GradientSparsification ([Deep gradient compression: Reducing the communication bandwidth for distributed training](https://openreview.net/forum?id=SkhQHMW0W))
  - This is a defense for active party 
  - "gradient_sparse_rate": ratio of gradients that will be set to zero, this value will be normalized in to $[0.0,1.0]$.

```json
"defense": {
        "name": "GradientSparsification",
        "parameters": {
            "gradient_sparse_rate": 100.0
        }
    }
```



## Standard Configuation Files

For quick and convenient usage, we provide several standard configuration files for different experiment settings in `/src/configs/standard_configs/` . 
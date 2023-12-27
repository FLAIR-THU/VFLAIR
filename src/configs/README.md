# How to write a configuration file

> This file offers a detailed tutorial for how to write a proper configuration file in VFLAIR to specify your own task.

[TOC]

### 1. Basic VFL Parameters

```json
"epochs": 30,
"lr": 0.005,
"k": 2,
"batch_size": 2048
```

- "epochs": number of iterations for each experiment
- "lr": main task learning rate
- "k": number of parties
  - While changing "k", model_list shall also be altered.
- "batch_size": #samples used for each batch



### 2. Dataset Parameters

```json
"dataset":{
        "dataset_name": "mnist",
        "num_classes": 10
    }
```

- "dataset": the dataset for experiments
  - "dataset_name": name of the dataset, ["mnist", "cifar10", "cifar100", "breast-cancer", "diabetes" ] are supported with default dataset partition, you can add your own dataset if needed by modifying `/src/load/LoadDataset.py` file. 
  - "num_classes": number of classes for experiments



### 3. Model Parameters

```json
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
        "apply_trainable_layer": 0,
        "global_model": "ClassificationModelHostHead"
}
```

- "model_list": specify all the models that are used in the experiment, should be a dictionary with all the party's index as keys
  - "i": for each party with index $i$
  - "type": local model type
  - "input_dim", "output_dim": necessary information to construct a local model
  - "path": specify trained model state dictionary if you want to load pre-trained models
- "apply_trainable_layer": whether the global model is trainable
- "global_model": global model



### 4. Communication Protocol Parameters

> e.g. CELU

```json
"communication":{
        "communication_protocol": "CELU",
        "iteration_per_aggregation": 5,
        "quant_level": 0,
        "vecdim": 1,
        "smi_thresh": 0.5
    }
```

- "communication_protocol": define the name of the communication protocal
- "iteration_per_aggregation": for [FedBCD](https://ieeexplore.ieee.org/abstract/document/9855231/), to increase VFL efficiency. When the value is set to `1`, then FedBCD is not applied, while for value `>1`, FedBCD is applied.



### 5. Attack & Defense

Your can load your own patameters for different attacks like the following example:

#### 5.1 Attack

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

#### 5.2 Defense

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
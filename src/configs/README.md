## README file for configurations

### One. Basic Parameters
* "epochs": number of iterations for each experiment
* "lr": main task learning rate
* "k": number of parties
* "batch_size": #samples used for each batch
* "iteration_per_aggregation": for [FedBCD](https://ieeexplore.ieee.org/abstract/document/9855231/), to increase VFL efficiency. When the value is set to `1`, then FedBCD is not applied, while for value `>1`, FedBCD is applied.
<!-- * "num_exp": experiments will repeat for this amount of times, and the average score is treated as the final result -->
* "dataset": the dataset for experiments
    * "dataset_name": name of the dataset, ["mnist", "cifar10", "cifar100", "breast-cancer", "diabetes" ] are supported with default dataset partition, you can add your own dataset if needed by modifying `/src/load/LoadDataset.py` file. 
    * "num_classes": number of classes for experiments
* "model_list": specify all the models that are used in the experiment, should be a dictionary with all the party's index as keys
    * "i": for each party with index $i$
        "type": local model type
        "input_dim", "output_dim": necessary information to construct a local model
        "path": specify trained model state dictionary if you want to load pre-trained models
    },
    * "apply_trainable_layer": whether the global model is trainable
    * "global_model": global model
* "attack": attack that is to be evaluated
    * "name": name for the attack, supported attacks list is shown in the bellow
    * "parameters": parameters for the specified attack
* "defense": defense that is to be evaluated
    * "name": name for the defense, supported defenses list is shown in the bellow
    * "parameters": parameters for the specified defense



### Two. Attack Parameters
* BatchLabelReconstruction ([Defending batch-level label inference and replacement attacks in vertical federated learning](https://ieeexplore.ieee.org/abstract/document/9833321/))
    * "party": list of party that launch this attack
    * "lr": learning rate for reconstruction model
    * "epochs": iterations for reconstruction model training
* ReplacementBackdoor ([Defending batch-level label inference and replacement attacks in vertical federated learning](https://ieeexplore.ieee.org/abstract/document/9833321/))
    * "party": list of parties that cooperate with each other to launch this attack
* For adding new attacks:
    * For a new inference time attack, first, implement this attack in `/src/evaluates/attacks/<your_attack>.py`; second, modify function `save_state` in `/src/evaluates/MainTaskVFL.py` to save necessary VFL information; finally add configurations in your config file, make sure the attack method has the same name to that in the config file. See `/src/evaluates/attacks/BatchLabelReconstruction.py` as an example.
    * For a new training time attack, implement another VFL flow with this attack, see `/src/evaluates/MainTaskVFLwithBackdoor.py` as an example.

### Two. Defense Parameters
* LaplaceDP ([Differential privacy: A survey of results](https://www.google.com.au/books/edition/Theory_and_Applications_of_Models_of_Com/JHFqCQAAQBAJ?hl=en&gbpv=1&pg=PA1&printsec=frontcover))
    * This is a defense for active party 
    * "dp_strength": laplace noise strength 0.001
* GaussianDP ([Differential privacy: A survey of results](https://www.google.com.au/books/edition/Theory_and_Applications_of_Models_of_Com/JHFqCQAAQBAJ?hl=en&gbpv=1&pg=PA1&printsec=frontcover))
    * This is a defense for active party 
    * "dp_strength": gaussian noise std 0.001
* GradientSparsification ([Deep gradient compression: Reducing the communication bandwidth for distributed training](https://openreview.net/forum?id=SkhQHMW0W))
    * This is a defense for active party 
    * "gradient_sparse_rate": ratio of gradients that will be set to zero, this value will be normalized in to $[0.0,1.0]$.
* CAE ([Defending batch-level label inference and replacement attacks in vertical federated learning](https://ieeexplore.ieee.org/abstract/document/9833321/))
    * This is a defense for active party 
    * "input_dim", "encode_dim": basic parameters for CAE encoder reconstruction
    * "lambda": $\lambda$, the confusional strength
    * "model_path": path of the trained CAE model, recomented model path is `/trained_CAE_models/<your_encoder>`
* MID ([Mutual Information Regularization for Vertical Federated Learning](https://arxiv.org/abs/2301.01142))
    * "party": list of parties that applies MID
    * "lr": learning rate for MID model
    * "lambda": $\lambda$, the hyper-parameter for MID, balancing between compression and information preservation
* For adding new defense:
    * If the defense is applied by adding new models, implement your new model in `/src/models/<your_model.py>` and then modify function `load_defense_models` in `/src/load/LoadModels.py` to add your model.
    * Otherwise, implement your own defense function in `/src/evaluates/defenses/defense_functions.py` and use `launch_defense` function of the VFL object in the proper place through out VFL main flow. See `launch_defense` function in `/src/evaluates/MainTaskVFL.py` as an example.
    * If the defense is difficult for implementation, implement another VFL flow with this defense, the same like implementing another VFL flow with a new attack.
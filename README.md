# VFLAIR

 ![Overview](usage_guidance/figures/overview.png)

## Basic Introduction

  VFLAIR is a general, extensible and light-weight VFL framework that provides vanilla VFL training and evaluation process simulation alonging with several effective communication improvement methods as well as attack and defense evaluations considering data safety and privacy. Aside from NN serving as local models for VFL systems, tree-based VFL is also supported.

VFLAIR provides simulation of the vanilla VFL process containing forward local model prediction transmits, backward gradient transmits as well as local and global model updates.


  * **Communication Library**: multiple communication protocols is provided for improving the effectiveness of VFL training process.
    * **FedBCD** ([paper](https://ieeexplore.ieee.org/abstract/document/9855231/)) 
    * **CELU** ([paper](https://arxiv.org/abs/2207.14628)) 
    * **Quantization** ([paper](https://arxiv.org/abs/2206.08330)) 
    * **Topk-Sparsification** ([paper](https://arxiv.org/abs/2206.08330)) 
  * **Attack&Defense Library**: 
      * Four attack types are included in VFLAIR as examples for training-decoupled attack and training-time attack separately. In each attack type, multiple attack is available for use:
        * **Label Inference(LI)** 
          * Batch-level Label Inference ([paper](https://ieeexplore.ieee.org/abstract/document/9833321/))/Direct Label Inference ([paper](https://ieeexplore.ieee.org/abstract/document/9833321/))
          * Norm-based Scoring (NS) ([paper]([[2102.08504\] Label Leakage and Protection in Two-party Split Learning (arxiv.org)](https://arxiv.org/abs/2102.08504)))/Direction-based Scoring (DS) ([paper]([[2102.08504\] Label Leakage and Protection in Two-party Split Learning (arxiv.org)](https://arxiv.org/abs/2102.08504)))
          * Passive Model Completion (PMC) ([paper]([Label Inference Attacks Against Vertical Federated Learning | USENIX](https://www.usenix.org/conference/usenixsecurity22/presentation/fu-chong)))/Active Model Completion (AMC) ([paper]([Label Inference Attacks Against Vertical Federated Learning | USENIX](https://www.usenix.org/conference/usenixsecurity22/presentation/fu-chong)))
        * **Feature Reconstruction(FR)**
          * Generative Regression Network (GRN)([paper]([[2010.10152\] Feature Inference Attack on Model Predictions in Vertical Federated Learning (arxiv.org)](https://arxiv.org/abs/2010.10152)))
          * Training-based Back Mapping by model inversion (TBM)([paper]([[2205.04007v1\] ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack in Split Federated Learning (arxiv.org)](https://arxiv.org/abs/2205.04007v1)))
        * **Targeted Backdoor(TB)**
          *  Label replacement Backdoor ([paper](https://ieeexplore.ieee.org/abstract/document/9833321/)) 
        * **Non-Targeted Backdoor(NTB)**
          * Noisy-Sample Backdoor (NSB)([paper](https://ieeexplore.ieee.org/abstract/document/9833321/)) 
          * Missing Feature (MF)([paper]([Liu2021.pdf (neurips2021workshopfl.github.io)](https://neurips2021workshopfl.github.io/NFFL-2021/papers/2021/Liu2021.pdf))) 
      * Several basic defense methods as well as emerging defense strategies are provided in VFLAIR and can be flexibly applied in VFL training and testing flow. Defense methods provided in VFLAIR is listed below. Detail information of these defenses are included in `/src/configs/README.md`.
        * **Differentail Privacy (Laplace-DP and Gaussian-DP)** ([paper](https://www.google.com.au/books/edition/Theory_and_Applications_of_Models_of_Com/JHFqCQAAQBAJ?hl=en&gbpv=1&pg=PA1&printsec=frontcover))
        * **Gradient Sparsification (GS)** ([paper](https://openreview.net/forum?id=SkhQHMW0W))
        * **Confusional AutoEncoder (CAE) & DiscreteSGD enhanced CAEï¼ˆDCAE)** ([paper](https://ieeexplore.ieee.org/abstract/document/9833321/))
        * **Mutual Information regularization Defense(MID)** ([paper](https://arxiv.org/abs/2301.01142))
        * **GradPerturb(GPer)** ([paper]([[2203.02073\] Differentially Private Label Protection in Split Learning (arxiv.org)](https://arxiv.org/abs/2203.02073)))
        * **Distance Correlation(dCor)** ([paper]([[2203.01451\] Label Leakage and Protection from Forward Embedding in Vertical Federated Learning (arxiv.org)](https://arxiv.org/abs/2203.01451)))
    * Defense Capability Score ——a comprehensive metric for assessing defense ability is also introduced.
  * **Dataset Library**: Multiple datasets are provided along with VFLAIR.
  * **Model Library**: Multiple models are provided along with VFLAIR and can be easily user-defined.
  * Other Functions
      * Tree-based VFL is also proved in the code with XGBoost and RandomForest supported. See `./src/configs/README_TREE.md` for detailed description. In adition, we currently support three defense methods against label leakage attack.
        * **LP-MST** ([paper](https://arxiv.org/abs/2102.06062))
        * **Grafting-LDP** ([paper](https://arxiv.org/abs/2307.10318))
        * **ID-LMID** ([paper](https://arxiv.org/abs/2307.10318))
    *  LLM-based VFL implementation is also provided with Bert, GPT2 and Llama supported. Please refer to `./src/configs/README_LLM.md` for detailed description.



## Code Structure

 ![Framework Structure](usage_guidance/figures/pipeline.png)

The VFLAIR framework is structured around four key functional modules:

- **Config Module:** Used for entering configuration parameters that define a VFL task.
- **Load Module:** Serves as a converter to translate configurations into actual code behavior. It is responsible for initializing datasets and parties (basic units of a VFL task).
- **Train & Evaluation Module:** Executes the final VFL training and model evaluation. It also incorporates user-defined attack/defense methods and communication protocols.
- **Metrics Module:** Converts raw experiment results into higher-level metrics such as DCS for benchmarking.

<!-- 
```
VFLAIR
├── src
│   ├── evaluates           
│   |   ├── attacks                    # Attack Simulator,Implementation of attacks
│   │   |   ├── ...                    # Multiple Attack Implementation
│   |   ├── defenses                   # Implementation of defenses
│   │   |   ├── Trained CAE momdels    # Trained encoder-decoder models for CAE and DCAE
│   │   |   ├── ...                    # Defense Implementation & Functions
│   |   ├── MainTaskVFL                # Pipeline for BasicVFL & VFL with LI/FR/NTB
│   |   ├── MainTaskVFLwithBackdoor    # Pipeline for VFL with TB     
│   |   ├── MainTaskVFLwithNoisySample # Pipeline for VFL with NTB-NSB    
│   |   ├── MainTaskTVFL               # Pipeline for Tree-based VFL
│   ├── load                           # Load Configurations into training pipeline
│   |   ├── LoadConfigs.py             # Load basic parameters   
│   |   ├── LoadDataset.py             # Load dataset and do data partition
│   |   ├── LoadModels.py              # Initialize models
│   |   ├── LoadParty.py               # Initialized parties with data and model
│   |   ├── LoadTreeConfigs.py         # Load basic parameters   
│   |   ├── LoadTreeParty.py           # Initialized parties with data and model
│   ├── configs                        # Customizable configurations    
│   |   ├── standard_configs           # Standard configurations for NN-based VFL
│   │   │   ├── ...   
│   |   ├── active_party_attack        # Standard configurations for active party attack
│   │   │   ├── ...   
│   |   ├── passive_party_attack       # Standard configurations for passive party attack
│   │   │   ├── ...   
│   |   ├── tree                       # Standard configurations for tree-based VFL 
│   │   │   ├── ...   
│   |   ├── README.md                  # Guidance for configuration files 
│   |   ├── README_TREE.md             # Guidance for testing tree-based VFL
│   ├── models                         # bottom models & global models     
│   |   ├── model_parameters           # Some pretrained models
│   │   ├── ...                        # Implemented bottome models & global models
│   ├── party                          # party simulators   
│   |   ├── ...
│   ├── dataset                        # Dataset preprocessing functions       
│   |   ├── ...
│   ├── utils                          # Basic functions and Customized functions for attack&defense
│   |   ├── ...
│   ├── exp_result                     # Store experiment results
│   |   ├── ...
│   ├── metrics                        # Benchmark and Defense Capability Score (DCS) definition
│   |   ├── ...
│   ├── main_pipeline.py               # Main VFL(launch this file for NN based VFL)  
│   ├── main_tree.py                   # Main Tree-based VFL(launch this file for tree-based VFL)  
├── usage_guidance                     # Detailed Usage  
│   ├── figures
│   |   ├── ...
│   ├── Add_New_Algorithm.md           # Guidance on how to add user defined attacks and defenses algorithms
│   ├── Dataset_Usage.md               # Guidance on how to achieve dataset for experiments
├── README.md
├── requirements.txt                   # installation requirement, we mainly use pytorch3.8 for experiments
``` -->


## Quick Start

### Zero. Environment Preparation

1. Download code files and install all the necessary requirements.

   ```shell
     # clone the repository
     $ git clone <link-to-our-github-repo>
     
     # install required packages
     $ conda create -n VFLAIR python=3.8
     $ conda activate VFLAIR
     $ pip install --upgrade pip
     $ cd VFLAIR
     $ pip install -r requirements.txt
     
     # install cuda related pytorch
     $ pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```

  2. New datasets, except MNIST, CIAFR10 and CIAFR100, can be put under `../../share_dataset/` folder on your device.

### One. Basic Benchmark Usage: A Quick Example

1. Customize your own configurations.
   * Create a json file for your own evaluation configuration in `/src/configs` folder. Name it whatever you want, like `my_configs.json`.
   * `/src/configs/basic_configs.json` is a sample configuration file. You can copy it and modify the contents for your own purpose.
   * For detail information about configuration parameters, see `/src/configs/README.md` for detail information.

2. Use `cd src` and `python main_pipeline.py --gpu 0 --configs <your_config_file_name>` to start the evaluation process. A quick example can be launched by simplying using `cd src` and `python main_pipeline.py` (a vanilla VFL training and testing process is launched). For more detail descriptions, see Section Two.

### Two. Advanced Usage: Implement Your Own Algorithm

- How to add new attack/defense?
  - `usage_guidance/Add_New_Evaluation.md`
- Dataset Usage?
  - `usage_guidance/Dataset_Usage.md`
- How to write Configuration files and how to specify hyper-parameters for evaluation?
  - `src/config/README.md` and `src/config/README_TREE.md`
- What is Defense Capability Score (DCS)?
  - Refer to `src/metrics` for details.




## Contributing

We **greatly appreciate** any contribution to VFLAIR! Also, we'll continue to improve our framework and documentation to provide more flexible and convenient usage.

Please feel free to contact us if there's any problem with the code base or documentation!

## Citation

If you are using VFLAIR for your work, please cite our paper with:

```
@article{zou2023vflair,
  title={VFLAIR: A Research Library and Benchmark for Vertical Federated Learning},
  author={Zou, Tianyuan and Gu, Zixuan and He, Yu and Takahashi, Hideaki and Liu, Yang and Ye, Guangnan and Zhang, Ya-Qin},
  journal={arXiv preprint arXiv:2310.09827},
  year={2023}
}
```
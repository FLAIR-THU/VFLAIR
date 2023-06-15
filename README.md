# VFLAIR

 ![Overview](usage_guidance/figures/overview.png)

## Basic Introduction

  VFLAIR is a general, extensible and light-weight VFL framework that provides vanilar VFL training and evaluation process simulation alonging with several effective improving methods as well as attack and defense evaluations considering data safety and privacy.

  * VFLAIR provides simulation of the vanilar VFL process containing forward local model prediction transmits, backward gradient transmits as well as local and global model updates.
  * **FedBCD** ([paper](https://ieeexplore.ieee.org/abstract/document/9855231/)) is provided for improving the effectiveness of VFL training process.
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
  * Multiple datasets are provided along with VFLAIR.
  * Defense Capability Score ——a comprehensive metric for assessing defense ability is also introduced.

**Batch-level Label Inference attack** ([paper](https://ieeexplore.ieee.org/abstract/document/9833321/)) and **targeted backdoor attack by label replacement** ([paper](https://ieeexplore.ieee.org/abstract/document/9833321/)) are included in VFLow as examples for training-decoupled attack and training-time attack separately. Detail information of these attacks are included in `/src/configs/README.md`.



## Code Structure

 ![VFLAIR](usage_guidance/figures/VFLAIR.png)


```
VFLow
├── src
│   ├── evaluates           
│   |   ├── attacks              # Attack Simulator,Implementation of attacks
│   |   ├── defenses             # Implementation of defenses
│   |   ├── MainTaskVFL          # Pipeline for BasicVFL & VFL with LI/FR/NTB
│   |   ├── MainTaskVFLwithBackdoor    # Pipeline for VFL with TB     
│   |   ├── ... ..
│   ├── load                      # Load Configurations into training pipeline
│   |   ├── LoadConfigs.py        # Load basic parameters   
│   |   ├── LoadDataset.py        # Load dataset and do data partition
│   |   ├── LoadModels.py         # Initialize models
│   |   ├── LoadParty.py          # Initialized parties with data and model
│   ├── configs                   # Customizable configurations          
│   ├── models                    # bottom models & global models     
│   ├── party                     # party simulator     
│   |   ├── party.py            
│   |   ├── active_party.py            
│   |   ├── passive_party.py
│   ├── dataset                   # Dataset preprocessing functions        
│   ├── utils                     # Basic functions
│   ├── exp_result                # Store experiment results
├── model parameters              # Some pretrained bottom models
├── Trained CAE momdels
├── Metrics                       # Benchmark and DCS definition
├── README.md
```



## Quick Start

### Zero. Environment Preparation

  Use `pip install -r requirements.txt` to install all the necessary requirements.

### One. Basic Benchmark Usage: A Quick Example

1. Customize your own configurations

* Create a json file for your own evaluation configuration in `/src/configs` folder. Name it whatever you want, like `my_configs.json`.
* `/src/configs/basic_configs.json` is a sample configuration file. You can copy it and modify the contents for your own purpose.
* For detail information about configuration parameters, see `/src/configs/README.md` for detail information.

2. Use `cd src` and `python main_separate.py --seed 0 --gpu 0 --configs <Your_Config_file_name>` to start the evaluation process. A quick example can be launched by simplying using `cd src` and `python main_separate.py` (a vanilar VFL training and testing process is launched). For more detail descriptions, see Section Two.

### Two. Advanced Usage: Implement Your Own Algorithm

- How to add new attack/defense?
  - `usage_guidance/Add_New_Evaluation.md`
- Dataset Usage?
  - `usage_guidance/Dataset_Usage.md`
- How to write Configuration files?
  - `src/config/README.md`
- What is DCS?
  - Refer to `src/metrics` for details.



## Document

### License



### Publications

If you find VFLAIR useful for your research or development, please cite as following:

```
@article{VFLAIR,
  title = {VFLAIR},
  author = {},
  year={2023}
}
```



## Contributing

We **greatly appreciate** any contribution to VFLAIR! 

Please feel free to contact us if there's any problem with the code base!
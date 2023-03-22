- # VFLow

## Basic Introduction

  VFLow is a general, extensible and light-weight VFL framework that provides vanilar VFL training and evaluation process simulation alonging with several effective improving methods as well as attack and defense evaluations considering data safety and privacy.

  * VFLow provides simulation of the vanilar VFL process containing forward local model prediction transmits, backward gradient transmits as well as local and global model updates.
  * **FedBCD** ([paper](https://ieeexplore.ieee.org/abstract/document/9855231/)) is provided for improving the effectiveness of VFL training process.
  * **Batch-level Label Inference attack** ([paper](https://ieeexplore.ieee.org/abstract/document/9833321/)) and **targeted backdoor attack by label replacement** ([paper](https://ieeexplore.ieee.org/abstract/document/9833321/)) are included in VFLow as examples for training-decoupled attack and training-time attack separately. Detail information of these attacks are included in `/src/configs/README.md`.
  * Several basic defense methods as well as emerging defense strategies are provided in VFLow and can be flexibly applied in VFL training and tesing flow. Defense methods provided in VFLow is listed below. Detail information of these defenses are included in `/src/configs/README.md`.
    * **Differentail Privacy (Laplace-DP and Gaussian-DP)** ([paper](https://www.google.com.au/books/edition/Theory_and_Applications_of_Models_of_Com/JHFqCQAAQBAJ?hl=en&gbpv=1&pg=PA1&printsec=frontcover))
    * **Gradient Sparsification (GS)** ([paper](https://openreview.net/forum?id=SkhQHMW0W))
    * **Confusional AutoEncoder (CAE) & DiscreteSGD enhanced CAE（DCAE)** ([paper](https://ieeexplore.ieee.org/abstract/document/9833321/))
    * **Mutual Information regularization Defense(MID)** ([paper](https://arxiv.org/abs/2301.01142))
  * Multipul datasets are provided along with VFLow. Detail introduction on dataset achival and application are explained below.

  

## Document
### Zero. Environment Preparation

  Use `pip install -r requirements.txt` to install all the necessary requirements.

### One. A quick example

  Use `cd src` and `python main_separate.py --seed 0 --gpu 0 --configs <Your_Config_file_name>` to start the evaluation process. A quick example can be launched by simplying using `cd src` and `python main_separate.py` (a vanilar VFL training and testing process is launched). For more detail descriptions, see Section Two.

### Two. How to use

  1. Run Evaluation Experiments
     * Create a json file for your own evaluation configuration in `/src/configs` folder. Name it whatever you want, like `my_configs.json`.
     * `/src/configs/basic_configs.json` is a sample configuration file. You can copy it and modify the contents for your own purpose.
     * For detail information about configuration parameters, see `/src/configs/README.md` for detail information.
  2. Add New Evaluations
     * If you want to add new attacks and defenses for evaluation, please refer to `/src/configs/README.md` for more details.
     * Other evluations can be added by starting from implementing your own algorithm. Then, you can modify `/src/load/` folder to prepare parameters, data, model etc. for your own algorithm. To implement them into the VFL flow, you can choose to create a new VFL flow based on `/src/evaluates/MainTaskVFL.py`.
       <!-- 2. Modify the configuration files for the attacks and defenses you want to use in `/src/configs/attacks` or `/src/configs/defenses`. Remeber to keep the configuration files' names as `<AttackName/DefenseName>_configs.json` and keep `<AttackName/DefenseName>` the same as the one in the json file you created in step1. -->
       <!-- 3. Add new functions for loading your config json file or dataset or model in `/src/load` directory and specify your data splition strategy.
  3. Prepare your model (the one that you want to evaluate) in `/model_parameters` folder. The program will load it via `pickle` format.
  4. Use `cd src` and `python main.py --seed 0 --gpu 0 --configs <Your_Config_file_name>` to start the evaluation process. -->

  <!-- ### Three. How to add new evaluation strategies

  1. Create your configuration file `<your_configs.json` for your evaluation in `/src/configs/` folder and add all the configuration parameters you need.
  2. Implement your loading function for your configurations in `/src/load/LoadConfigs.py`.
  3. Prepare your dataset if it has not been provied. Put it in `/data` folder. And prepare the data loading function in `/src/load/LoadDataset.py`. Also if you want to use your own data splitting strategy which is not yet supported, add it in `/src/load/LoadDataset.py` as well.
  4. Prepare your own model in `/model_parameters` as pickle files. Simply use `pickle.dump(your_net, open('<YourPath>/<YourModel>.pkl','wb'))` to save your model and use `your_net = pickle.load(open('<YourPath>/<YourModel>.pkl',"rb"))` to load it.
  5. Add the name of your attack(defense) method to `/src/configs/basic_configs.json` file and set the value to `1` if you want it in your evaluation. -->

### Three. Datasets

  In VFLow, we defaultly provide several dataset that are oftenly used under VFL setting. As the origianl dataset is not provided along with the codebase, all the dataset provided and guidance on how to achieve these datasets are listed below.

  Defaultly, datasets used are stored under directory `../../share_dataset/`.

  1. Image Dataset

     1. MNIST
        * Loaded from torchvision, `torchvision.datasets.MNIST('~/.torch', download=True)`
     2. CIFAR10
        * By default, the data are stored under directory `../../share_dataset/cifar-10-batches-py/`.
        * Loaded from torchvision, `torchvision.datasets.CIFAR10('<you_directory>', download=True)`
     3. CIFAR100
        * By default, the data are stored under directory `../../share_dataset/cifar-100-python/`.
        * Loaded from torchvision, `torchvision.datasets.CIFAR100('<you_directory>', download=True)`

  2. Graph Dataset

     1. Cora
        * By default, the data are stored under directory `../../share_dataset/Cora/`.
        * We use the dataset provided by [Cora (Github)](https://github.com/hgh0545/Graph-Fraudster) for Cora.

  3. Tabular Dataset

     1. Breast Cancer

       * By default, the data are stored under directory `../../share_dataset/BreastCancer/`
       * Download wdbc.data from [Wisconsin Diagnostic Breast Cancer (WDBC) | Kaggle](https://www.kaggle.com/datasets/mohaiminul101/wisconsin-diagnostic-breast-cancer-wdbc)

     2. Diabetes
        * By default, the data are stored under directory `../../share_dataset/Diabetes/`
        * Download diabetes.csv from [Pima Indians Diabetes | Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
     3. Adult Income
        * By default, the data are stored under directory `../../share_dataset/Income/`
        * Download adult.csv from [Adult income dataset | Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)
      4. Cretio
         * By default, the data are stored under directory `../../share_dataset/Criteo/`.
         * Download `tain.txt` from [Criteo | Kaggle](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset) and run `python /src/dataset/criteo_preprocess.py` to create `criteo.csv` in `../../share_dataset/Criteo/`.

  4. Text Dataset

     1. News20
        * By default, the data are stored under directory `../../share_dataset/news20/`
        * Download from [20 Newsgroups]([Home Page for 20 Newsgroups Data Set (qwone.com)](http://qwone.com/~jason/20Newsgroups/)), and align texts from the same category into the same directory as`../../share_dataset/news20/"category_name"/text_files`
        * 3 versions of the news20 dataset are available(20news-19997.tar.gz/20news-bydate/tar.gz/20news-18828.tar.gz). In VFLow, we use 20news-19997.tar.gz by default.
        * TF-IDF is used for data processing, turning each text into a sparse matrix. Dimension of the matrix may vary using different versions of the news20 dataset, therefor affecting the 'input_dim' in bottom models. Please refer to [] for details. 

  5. Multi-modal Dataset

     1. NUS-WIDE
        * By default, the data are stored under directory `../../share_dataset/NUS_WIDE/`.
        * Download from [NUS-WIDE Dataset](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html), only `Groundtruth, Low_level_Features, NUS_WIDE_Tags, NUS_WIDE-urls` are needed in this project.



### Four. Function Usage & Parameters

#### (1) Basic VFL

- Supported Datasets: 

  MNIST(Default) / CIFAR10 / CIFAR100 / Cora / Breast Cancer / Diabetes / Adult Income / Cretio / News20 / NUS-WIDE

|                Parameters                 |       Allowed        | Recommended |                      Description                      |
| :---------------------------------------: | :------------------: | :---------: | :---------------------------------------------------: |
|           Party number : args.k           |        2 / 4         |      2      | NOTE ：k=4 only available for MNIST/CIFAR10/CIFAR100  |
| FedBCD : args.Q Iteration_per_aggregation | any positive integer |      3      |     Iteration_per_aggregation in FedBCD algorithm     |
|        args.apply_trainable_layer         |        0 / 1         |      1      | Whether to apply one trainable layer in the top model |



#### (2) Attack & Defense

- Supported Datasets: 

  MNIST / CIFAR10 / CIFAR100

#### Attack

##### BatchLabelReconstruction

| Parameters | Recommended |                         Description                          |
| :--------: | :---------: | :----------------------------------------------------------: |
|     lr     |    0.05     | Learning rate in the label inference model in passive parties |
|   epochs   |    10000    |                                                              |
|   party    |     [0]     |        Attack is launched by passive party (party[0])        |

##### ReplacementBackdoor

| Parameters | recommended |                  Description                   |
| :--------: | :---------: | :--------------------------------------------: |
|   party    |     [0]     | Attack is launched by passive party (party[0]) |

#### Defense

|         Defense         |      parameters      | recommended |            Description            |
| :---------------------: | :------------------: | :---------: | :-------------------------------: |
|   Laplace/Gaussian DP   |     dp_strength      |    0.001    | Control the strength on the noise |
| Gradient Sparsification | gradient_sparse_rate |     100     |                                   |
|           CAE           |        lambda        |      1      |                                   |
|                         |      model_path      |      -      |  path for pretrained AutoEncoder  |
|           MID           |        lambda        |      0      |                                   |
|                         |          lr          |    0.01     |                                   |
|                         |        party         |     [1]     |                                   |

- CAE and MID often gain better performance than Laplace/Gaussian DP and Gradient Sparsification. ( Need to add evaluation folder for reference)




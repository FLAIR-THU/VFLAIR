#### ToDo List
- [x] requirement list generation. @tianyuan
- [ ] Attacks and Defense to add @tianyuan
- [ ] FedBCD to add @zixuan
- [x] Logic of attack and defense needs update @tianyuan


## Document

### Zero. Environment Preparation
Use `pip install -r requirements.txt` to install all the necessary requirements.

### One. A quick example
Use `cd src` and `python main_separate.py --seed 0 --gpu 0 --configs <Your_Config_file_name>` to start the evaluation process. A quick example can be launched by simplying using `cd src` and `python main_separate.py`. For more detail description, see Section Two.

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
4. Prepare your model (the one that you want to evaluate) in `/model_parameters` folder. The program will load it via `pickle` format.
5. Use `cd src` and `python main.py --seed 0 --gpu 0 --configs <Your_Config_file_name>` to start the evaluation process. -->

<!-- ### Three. How to add new evaluation strategies
1. Create your configuration file `<your_configs.json` for your evaluation in `/src/configs/` folder and add all the configuration parameters you need.
3. Implement your loading function for your configurations in `/src/load/LoadConfigs.py`.
4. Prepare your dataset if it has not been provied. Put it in `/data` folder. And prepare the data loading function in `/src/load/LoadDataset.py`. Also if you want to use your own data splitting strategy which is not yet supported, add it in `/src/load/LoadDataset.py` as well.
5. Prepare your own model in `/model_parameters` as pickle files. Simply use `pickle.dump(your_net, open('<YourPath>/<YourModel>.pkl','wb'))` to save your model and use `your_net = pickle.load(open('<YourPath>/<YourModel>.pkl',"rb"))` to load it.
6. Add the name of your attack(defense) method to `/src/configs/basic_configs.json` file and set the value to `1` if you want it in your evaluation. -->

# Three. Datasets
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
        * We use the dataset provided by [link](https://github.com/hgh0545/Graph-Fraudster) for Cora.
3. Tabular Dataset
    1. Breast Cancer
      * By default, the data are stored under directory `../../share_dataset/BreastCancer/`
      * Download wdbc.data from [Wisconsin Diagnostic Breast Cancer (WDBC) | Kaggle](https://www.kaggle.com/datasets/mohaiminul101/wisconsin-diagnostic-breast-cancer-wdbc)
   2. Diabetes
      * By default, the data are stored under directory `../../share_dataset/Diabetes/`
      * Download diabetes.csv from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
   3. Adult Income
      * By default, the data are stored under directory `../../share_dataset/Income/`
      * Download adult.csv from [Adult income dataset | Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)
    4. Cretio
        * By default, the data are stored under directory `../../share_dataset/Criteo/`.
        * Download `tain.txt` from [kaggle-link](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset) and run `python /src/dataset/criteo_preprocess.py` to create `criteo.csv` in `../../share_dataset/Criteo/`.
4. Text Dataset
    1. News20
        * By default, the data are stored under directory `../../share_dataset/news20/`
        * Download from [20 Newsgroups | Kaggle](https://www.kaggle.com/datasets/crawford/20-newsgroups), and align texts from the same category into the same directory as`../../share_dataset/news20/"category_name"/text_files`
5. Multi-modal Dataset
    1. NUS-WIDE
        * By default, the data are stored under directory `../../share_dataset/NUS_WIDE/`.
        * Download from [link](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html), only `Groundtruth, Low_level_Features, NUS_WIDE_Tags, NUS_WIDE-urls` are needed in this project.

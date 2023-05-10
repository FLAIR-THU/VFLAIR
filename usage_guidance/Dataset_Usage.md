# Datasets

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


#### ToDo List
- [ ] base_config.json and \<DefenseMethod\>_confige.json @tianyuan
- [ ] BatchLabelReconstruction change the models to list (multi-party?) @tianyuan
- [ ] Attacks and Defense to add @tianyuan
- [ ] FedBCD to add @tianyuan
- [ ] Logic of attack and defense needs update @tianyuan


## Document

### One. A quick example
Just run `python main_separate.py` to get launch an experiment. For more detail description, see Section Two.

### Two. How to use
1. Create a json file for your own evaluation configuration in `/src/configs` folder. Name it whatever you want, like `my_configs.json`. `/src/configs/basic_configs.json` is a sample configuration file. You can copy it and modify the contents for your own purpose.
2. Modify the configuration files for the attacks and defenses you want to use in `/src/configs/attacks` or `/src/configs/defenses`. Remeber to keep the configuration files' names as `<AttackName/DefenseName>_configs.json` and keep `<AttackName/DefenseName>` the same as the one in the json file you created in step1.
3. Add new functions for loading your config json file or dataset or model in `/src/load` directory and specify your data splition strategy.
4. Prepare your model (the one that you want to evaluate) in `/model_parameters` folder. The program will load it via `pickle` format.
5. Use `cd src` and `python main.py --seed 0 --gpu 0 --configs <Your_Config_file_name>` to start the evaluation process.

### Three. How to add new evaluation strategies
1. Implement your attack(denfese) algorithm in `/src/evaluates(defenses)/<YourAttack(YourDefense)>.py` and name the function as `<YorAttack(YourDefense)>`.
    * Don't forget to add entries for defense methods in all the attacks' function that you want to defend.
2. Create the configuration file `<YourAttack(YourDefense)_configs.json` for your attack(denfese) algorithm in `/src/configs/attacks(defenses)` folder and add all the configuration parameters you need.
3. Implement your loading function for your configurations in `/src/load/LoadConfigs.py`.
4. Prepare your dataset if it has not been provied. Put it in `/data` folder. And prepare the data loading function in `/src/load/LoadDataset.py`. Also if you want to use your own data splitting strategy which is not yet supported, add it in `/src/load/LoadDataset.py` as well.
5. Prepare your own model in `/model_parameters` as pickle files. Simply use `pickle.dump(your_net, open('<YourPath>/<YourModel>.pkl','wb'))` to save your model and use `your_net = pickle.load(open('<YourPath>/<YourModel>.pkl',"rb"))` to load it.
6. Add the name of your attack(defense) method to `/src/configs/basic_configs.json` file and set the value to `1` if you want it in your evaluation.

### Four. Currently supporting attacks and defenses lists (Future Plane, not done)
0. Basic VFL
    1. MainTaskVFL_separate.py support basic VFL with 2-party
1. Attacks
    1. Batch-level Label Inference Attack
    2. Sample-level Label Inference Attack
    3. 
2. Defenses
    1. Differencial Privacy with Laplace Noise
    2. Differencial Privacy with Gaussian Noise
    3. Gradient Sparcification
    4. Discrete Gradient (Discrete SGD)
    5. Confusional AutoEncoder
    6. Discrete Confusional AutoEncoder
    7. Marvell
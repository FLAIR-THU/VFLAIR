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


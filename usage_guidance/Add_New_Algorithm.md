# Add New Algorithms
## How To Add New Evaluation
    1. Add New Evaluations
       * If you want to add new attacks and defenses for evaluation, please refer to `/src/configs/README.md` for more details.
       * Other evluations can be added by starting from implementing your own algorithm. Then, you can modify `/src/load/` folder to prepare parameters, data, model etc. for your own algorithm. To implement them into the VFL flow, you can choose to create a new VFL flow based on `/src/evaluates/MainTaskVFL.py`.
         <!-- 2. Modify the configuration files for the attacks and defenses you want to use in `/src/configs/attacks` or `/src/configs/defenses`. Remeber to keep the configuration files' names as `<AttackName/DefenseName>_configs.json` and keep `<AttackName/DefenseName>` the same as the one in the json file you created in step1. -->
         <!-- 3. Add new functions for loading your config json file or dataset or model in `/src/load` directory and specify your data splition strategy.
    2. Prepare your model (the one that you want to evaluate) in `/model_parameters` folder. The program will load it via `pickle` format.
    3. Use `cd src` and `python main.py --seed 0 --gpu 0 --configs <Your_Config_file_name>` to start the evaluation process. -->
      <!-- ### Three. How to add new evaluation strategies

    1. Create your configuration file `<your_configs.json` for your evaluation in `/src/configs/` folder and add all the configuration parameters you need.
    2. Implement your loading function for your configurations in `/src/load/LoadConfigs.py`.
    3. Prepare your dataset if it has not been provied. Put it in `/data` folder. And prepare the data loading function in `/src/load/LoadDataset.py`. Also if you want to use your own data splitting strategy which is not yet supported, add it in `/src/load/LoadDataset.py` as well.
    4. Prepare your own model in `/model_parameters` as pickle files. Simply use `pickle.dump(your_net, open('<YourPath>/<YourModel>.pkl','wb'))` to save your model and use `your_net = pickle.load(open('<YourPath>/<YourModel>.pkl',"rb"))` to load it.
    5. Add the name of your attack(defense) method to `/src/configs/basic_configs.json` file and set the value to `1` if you want it in your evaluation. -->

## How To Add New Attack&Defense
1. For adding new attacks
- For a new inference time attack, first, implement this attack in `/src/evaluates/attacks/<your_attack>.py`; second, modify function `save_state` in `/src/evaluates/MainTaskVFL.py` to save necessary VFL information; finally add configurations in your config file, make sure the attack method has the same name to that in the config file. See `/src/evaluates/attacks/BatchLabelReconstruction.py` as an example.
- For a new training time attack, implement another VFL flow with this attack, see `/src/evaluates/MainTaskVFLwithBackdoor.py` as an example.
2. For adding new defense:
- If the defense is applied by adding new models, implement your new model in `/src/models/<your_model.py>` and then modify function `load_defense_models` in `/src/load/LoadModels.py` to add your model.
- Otherwise, implement your own defense function in `/src/evaluates/defenses/defense_functions.py` and use `launch_defense` function of the VFL object in the proper place through out VFL main flow. See `launch_defense` function in `/src/evaluates/MainTaskVFL.py` as an example.
- If the defense is difficult for implementation, implement another VFL flow with this defense, the same like implementing another VFL flow with a new attack.

# Add New Algorithms

## How To Add New Evaluation In General

1. Add New Evaluations

   * Modify `/src/load/` folder to **prepare parameters, data, model** etc. for your own algorithm. 
     * Prepare your model (the one that you want to evaluate) in `/model_parameters` folder. The program will load it via `pickle` format.
     * In you need to use your own model, prepare them in `/model_parameters` as pickle files. Simply use `pickle.dump(your_net, open('<YourPath>/<YourModel>.pkl','wb'))` to save your model and use `your_net = pickle.load(open('<YourPath>/<YourModel>.pkl',"rb"))` to load it.
   * To implement them into the VFL flow, you can choose to create **a new VFL flow** based on `/src/evaluates/MainTaskVFL.py`.
   * You can **add processing functions** in `/src/utils/`,`/src/evaluate/attacks/` and `/src/evaluate/defenses/` according to your needs.

2. Load Configurations

   - Implement your **loading function** for your configurations in `/src/load/LoadConfigs.py`.

   - Create your **configuration file** `<your_configs.json` for your evaluation in `/src/configs/` folder and add all the configuration parameters you need.

3. Finally, Use `cd src` and `python main.py --seed 0 --gpu 0 --configs <Your_Config_file_name>` to start the evaluation process. -->



## How To Add New Attack&Defense

1. For adding new attacks

- For a new inference time attack, first, implement this attack in `/src/evaluates/attacks/<your_attack>.py`; second, modify function `save_state` in `/src/evaluates/MainTaskVFL.py` to save necessary VFL information; finally add configurations in your config file, make sure the attack method has the same name to that in the config file. See `/src/evaluates/attacks/BatchLabelReconstruction.py` as an example.
- For a new training time attack, implement another VFL flow with this attack, see `/src/evaluates/MainTaskVFLwithBackdoor.py` as an example.

2. For adding new defense:

- If the defense is applied by adding new models, implement your new model in `/src/models/<your_model.py>` and then modify function `load_defense_models` in `/src/load/LoadModels.py` to add your model.
- Otherwise, implement your own defense function in `/src/evaluates/defenses/defense_functions.py` and use `launch_defense` function of the VFL object in the proper place through out VFL main flow. See `launch_defense` function in `/src/evaluates/MainTaskVFL.py` as an example.
- If the defense is difficult for implementation, implement another VFL flow with this defense, the same like implementing another VFL flow with a new attack.
import sys, os

sys.path.append(os.pardir)

from evaluates.defenses.defense_functions import *


def DefenderLoader(args, index):
    if args.apply_defense == False:
        return None
    elif args.defense_name in ['MID']:
        # for future use, create a defender
        return None
    elif args.defense_name in ['LaplaceDP', 'GaussianDP', 'GradientSparsification', 'DiscreteGradient', 'MARVELL',
                               'CAE', 'DCAE', 'DiscreteSGD']:
        # simple function, no need for defender at class
        return None


def apply_defense(args, _type, *params):
    # LLM scenario
    if args.model_type != None:  
        if args.defense_name in ['LaplaceDP', 'GaussianDP']:
            defense_name = args.defense_name + '_for_llm'
            pred_list = params
            return globals()[defense_name](args, pred_list)
    # Normal VFL
    else:  
        if _type == "gradients":
            if args.defense_name in ['LaplaceDP', 'GaussianDP', 'GradientSparsification', 'DiscreteSGD', 'GradPerturb']:
                gradient_list = params
                return globals()[args.defense_name](args, gradient_list)
            elif args.defense_name in ['DCAE']:
                gradient_list = params
                return globals()['DiscreteSGD'](args, gradient_list)
        elif _type == "pred":
            if args.defense_name in ['LaplaceDP', 'GaussianDP']:
                defense_name = args.defense_name + '_for_pred'
                pred_list = params
                return globals()[defense_name](args, pred_list)

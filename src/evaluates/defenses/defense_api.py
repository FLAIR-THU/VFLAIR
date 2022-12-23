import sys, os
sys.path.append(os.pardir)

from evaluates.defenses.defense_functions import *

def DefenderLoader(args, index):
    if args.apply_defense == False:
        return None
    elif args.defense_name in ['MID']:
        # for future use, create a defender
        return None
    elif args.defense_name in ['LaplaceDP', 'GaussianDP', 'GradientSparsification', 'DiscreteGradient', 'MARVELL', 'CAE', 'DCAE']:
        # simple function, no need for defender at class
        return None

def apply_defense(args, *params):
    if args.defense_name in ['LaplaceDP', 'GaussianDP', 'GradientSparsification']:
        gradient_list = params
        return globals()[args.defense_name](args, gradient_list)

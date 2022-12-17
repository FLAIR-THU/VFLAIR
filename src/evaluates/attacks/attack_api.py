import sys, os
sys.path.append(os.pardir)

from evaluates.attacks.BatchLabelReconstruction import BatchLabelReconstruction

# def apply_attack(args, gradient_list, pred_list, type):
#     if type == 'gradients':
#         pass

# def AttackerLoader(args, index, local_model):
#     attacker_name = args.attack_name
#     return globals()[attacker_name](args, index, local_model)
def AttackerLoader(vfl, args):
    attacker_name = args.attack_name
    return globals()[attacker_name](vfl, args)
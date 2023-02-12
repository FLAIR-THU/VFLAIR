import sys, os
sys.path.append(os.pardir)

from evaluates.attacks.BatchLabelReconstruction import BatchLabelReconstruction

def AttackerLoader(vfl, args):
    attacker_name = args.attack_name
    return globals()[attacker_name](vfl, args)
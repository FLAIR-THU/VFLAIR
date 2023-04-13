import sys, os
sys.path.append(os.pardir)

from evaluates.attacks.BatchLabelReconstruction import BatchLabelReconstruction
from evaluates.attacks.DataReconstruct import DataReconstruction
from evaluates.attacks.DirectionbasedScoring import DirectionbasedScoring
from evaluates.attacks.NormbasedScoring import NormbasedScoring
from evaluates.attacks.NoisyLabel import NoisyLabel

def AttackerLoader(vfl, args):
    attacker_name = args.attack_name
    if attacker_name == "DataLabelReconstruction":
        assert args.batch_size == 1,'DataLabelReconstruction: require batchsize=1'
        attacker_name == "BatchLabelReconstruction"
    return globals()[attacker_name](vfl, args)
import sys, os
sys.path.append(os.pardir)


def AttackerLoader(vfl, args):
    attacker_name = args.attack_name
    if 'ModelCompletion' in attacker_name:
        attacker_name = 'ModelCompletion'
    # if attacker_name == "DataLabelReconstruction":
    #     assert args.batch_size == 1,'DataLabelReconstruction: require batchsize=1'
    #     attacker_name == "BatchLabelReconstruction"
    return globals()[attacker_name](vfl, args)
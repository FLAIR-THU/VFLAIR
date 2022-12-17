import sys, os
sys.path.append(os.pardir)


# class Attacker(object):
#     def __init__(self, args, index, local_model):
#         self.name = args.attack_name
#         self.parameters = args.attack_configs
#         self.attacker_party_local_model = local_model
    
#     def attack(self, *params):
#         pass

class Attacker(object):
    def __init__(self, args):
        self.name = args.attack_name
        self.parameters = args.attack_configs
    
    def attack(self, *params):
        pass



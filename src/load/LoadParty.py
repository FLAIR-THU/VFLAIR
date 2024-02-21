import sys, os
sys.path.append(os.pardir)

from party.passive_party import PassiveParty
from party.active_party import ActiveParty
from party.paillier_passive_party import PaillierPassiveParty
from party.paillier_active_party import PaillierActiveParty

def load_parties(args):
    # party 0,1,2,...,args.k-2||,args,k-1
    args.parties = [None] * args.k 

    assert args.k >= 1
    # for passive party 0,1,2,...,args.k-2
    for ik in range(args.k-1):
        args.parties[ik] = PassiveParty(args, ik)

    # for active party args.k-1
    args.parties[args.k-1] = ActiveParty(args, args.k-1)

    return args

def load_parties_llm(args):
    # party 0,1,2,...,args.k-2||,args,k-1
    args.parties = [None] * args.k 

    assert args.k >= 1
    # for passive party 0,1,2,...,args.k-2
    for ik in range(args.k-1):
        args.parties[ik] = PassiveParty_LLM(args, ik)
    # for active party args.k-1
    args.parties[args.k-1] = ActiveParty_LLM(args, args.k-1)

    # # for server party 0,1,2,...,args.k_server-1
    # for ik in range(args.k_server):
    #     args.servers[ik] = Server(args, ik)

    return args


def load_paillier_parties(args, pk, sk):
    # party 0,1,2,...,args.k-2||,args,k-1
    args.parties = [None] * args.k 

    assert args.k >= 1
    # for passive party 0,1,2,...,args.k-2
    for ik in range(args.k-1):
        args.parties[ik] = PaillierPassiveParty(args, ik)
        args.parties[ik].set_keypairs(pk, sk)

    # for active party args.k-1
    args.parties[args.k-1] = PaillierActiveParty(args, args.k-1)

    return args

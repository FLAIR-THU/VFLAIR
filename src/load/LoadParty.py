import sys, os

sys.path.append(os.pardir)

from party.passive_party import PassiveParty
from party.active_party import ActiveParty
from party.passive_party import PassiveParty_LLM
from party.active_party import ActiveParty_LLM
from party.qwen_active_party import QW_Active_Party
from party.qwen_passive_party import QW_Passive_Party
from party.paillier_passive_party import PaillierPassiveParty
from party.paillier_active_party import PaillierActiveParty


def load_parties(args):
    # party 0,1,2,...,args.k-2||,args,k-1
    args.parties = [None] * args.k

    assert args.k >= 1
    # for passive party 0,1,2,...,args.k-2
    for ik in range(args.k - 1):
        args.parties[ik] = PassiveParty(args, ik)

    # for active party args.k-1
    args.parties[args.k - 1] = ActiveParty(args, args.k - 1)

    return args


def get_class_constructor(class_name):
    return globals()[class_name]


def load_parties_llm(args, need_data=True):
    # party 0,1,2,...,args.k-2||,args,k-1
    args.parties = [None] * args.k

    assert args.k >= 1
    # for passive party 0,1,2,...,args.k-2
    for ik in range(args.k - 1):
        args.parties[ik] = get_class_constructor(args.passive_party_class)(args, ik, need_data=need_data)
    # for active party args.k-1
    args.parties[args.k - 1] = get_class_constructor(args.active_party_class)(args, args.k - 1, need_data=need_data)

    # # for server party 0,1,2,...,args.k_server-1
    # for ik in range(args.k_server):
    #     args.servers[ik] = Server(args, ik)

    return args


def load_paillier_parties(args, pk):
    # party 0,1,2,...,args.k-2||,args,k-1
    args.parties = [None] * args.k

    assert args.k >= 1
    # for passive party 0,1,2,...,args.k-2
    for ik in range(args.k - 1):
        args.parties[ik] = PaillierPassiveParty(args, ik)
        args.parties[ik].set_pk(pk)

    # for active party args.k-1
    args.parties[args.k - 1] = PaillierActiveParty(args, args.k - 1)

    return args

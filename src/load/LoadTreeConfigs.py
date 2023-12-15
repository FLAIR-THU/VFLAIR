import argparse
import json
import sys
import os

sys.path.append(os.pardir)


def load_tree_configs(config_file_name, args):
    config_file_path = "./configs/" + config_file_name + ".json"
    config_file = open(config_file_path, "r")
    config_dict = json.load(config_file)

    args.k = config_dict["k"] if ("k" in config_dict) else 2
    args.active_party_id = (
        config_dict["active_party_id"] if ("active_party_id" in config_dict) else 0
    )
    args.model_type = config_dict["tree_type"]
    args.number_of_trees = (
        config_dict["number_of_trees"] if ("number_of_trees" in config_dict) else 3
    )
    args.depth = config_dict["depth"] if ("depth" in config_dict) else 2
    args.min_leaf = config_dict["min_leaf"] if ("min_leaf" in config_dict) else 1
    args.subsample_cols = (
        config_dict["subsample_cols"] if ("subsample_cols" in config_dict) else 0.8
    )
    args.max_bin = config_dict["max_bin"] if ("max_bin" in config_dict) else 4
    args.advanced_params = (
        config_dict["advanced_params"] if ("advanced_params" in config_dict) else {}
    )
    args.key_length = (
        config_dict["key_length"] if ("key_length" in config_dict) else 128
    )
    args.use_missing_value = (
        config_dict["use_missing_value"]
        if ("use_missing_value" in config_dict)
        else False
    )
    args.is_hybrid = config_dict["is_hybrid"] if ("is_hybrid" in config_dict) else False
    args.use_encryption = (
        config_dict["use_encryption"] if ("use_encryption" in config_dict) else False
    )
    args.use_encrypted_label = (config_dict["use_encrypted_label"] if ("use_encrypted_label" in config_dict) else False)

    args.apply_defense = "defense" in config_dict

    if not args.apply_defense:
        config_dict["defense"] = {}

    args.defense_name = (
        config_dict["defense"]["name"] if ("name" in config_dict["defense"]) else None
    )

    if "parameters" not in config_dict["defense"]:
        config_dict["defense"]["parameters"] = {}

    args.lpmst_eps = (
        config_dict["defense"]["parameters"]["lpmst_eps"]
        if ("lpmst_eps" in config_dict["defense"]["parameters"])
        else 1.0
    )
    args.lpmst_m = (
        config_dict["defense"]["parameters"]["lpmst_m"]
        if ("lpmst_m" in config_dict["defense"]["parameters"])
        else 2
    )
    args.mi_bound = (
        config_dict["defense"]["parameters"]["mi_bound"]
        if ("mi_bound" in config_dict["defense"]["parameters"])
        else -1.0
    )

    return args

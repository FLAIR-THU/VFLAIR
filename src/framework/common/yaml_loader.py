"""A simple yaml loader
    Load the content of an yaml file into a python dict. Environment variable can be specified with ${VAR_NAME}. A
    default string value can be specified with ${VAR_NAME:DEFAULT}. Able to process multiple occurrences.
    requirement: PyYAML
    run: python loader.py
"""

import os
import re
from typing import Any

import yaml
from yaml.parser import ParserError

_var_matcher = re.compile(r"\${([^}^{]+)}")
_tag_matcher = re.compile(r"[^$]*\${([^}^{]+)}.*")


def _path_constructor(_loader: Any, node: Any):
    def replace_fn(match):
        envparts = f"{match.group(1)}:".split(":")
        return os.environ.get(envparts[0], envparts[1])

    return _var_matcher.sub(replace_fn, node.value)


def load_yaml(filename: str) -> dict:
    yaml.add_implicit_resolver("!envvar", _tag_matcher, None, yaml.SafeLoader)
    yaml.add_constructor("!envvar", _path_constructor, yaml.SafeLoader)
    try:
        with open(filename, "r") as f:
            return yaml.safe_load(f.read())
    except (FileNotFoundError, PermissionError, ParserError):
        return dict()

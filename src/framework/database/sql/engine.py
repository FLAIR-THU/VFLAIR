from sqlmodel import SQLModel, create_engine
from framework.common.yaml_loader import load_yaml
import os

from framework.database.model.Task import Task
from framework.database.model.Job import Job
from framework.database.model.PretrainedModel import PretrainedModel


def load():
    dir = os.path.dirname(__file__)
    config = load_yaml(dir + "/../../client/client_config.yml")
    database = config.get("database")
    return create_engine(database["url"], echo=False)


engine = load()
SQLModel.metadata.create_all(engine)

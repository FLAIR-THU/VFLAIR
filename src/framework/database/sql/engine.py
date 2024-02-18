from sqlmodel import SQLModel, create_engine
from framework.common.yaml_loader import load_yaml
import os

from framework.database.model.Task import Task
from framework.database.model.Job import Job


def load():
    dir = os.path.dirname(__file__)
    config = load_yaml(dir + "/../../server/server_config.yml")
    database = config.get("database")
    return create_engine(database["url"], echo=True)


engine = load()
SQLModel.metadata.create_all(engine)

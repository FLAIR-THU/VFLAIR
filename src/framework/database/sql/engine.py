from datetime import datetime

from sqlmodel import SQLModel, create_engine, Session
import yaml
import os

from framework.database.model.Task import Task
from framework.database.model.Job import Job

def load():
    dir = os.path.dirname(__file__)
    config = yaml.safe_load(open(dir + "/../../server/server_config.yml"))
    database = config.get("database")
    return create_engine(database["url"], echo=True)


engine = load()
SQLModel.metadata.create_all(engine)

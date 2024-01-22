import argparse
from fastapi import FastAPI, Form
import uvicorn

from load.LoadConfigs import load_llm_configs
from framework.client.grpc_client import GrpcClient
import framework.common.MessageUtil as mu
import framework.protos.node_pb2 as fpn
import framework.protos.message_pb2 as fpm
from typing_extensions import Annotated
import json
import yaml
import framework.common.logger_util as logger_util
from contextlib import asynccontextmanager

service = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    args = parse_args()
    print(args)
    init_grpc_client(args)
    yield
    # Clean up the ML models and release the resources
    service.clear()

app = FastAPI(lifespan=lifespan)

node = fpn.Node(node_id="web")

logger = logger_util.get_logger("web_server")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/job")
def create_job(config: Annotated[str, Form()]):
    data = load_llm_configs(config, argparse.Namespace())
    if (data.fl_type == 'VFL'):
        value = fpm.Value()
        value.string = json.dumps(vars(data))
        msg = mu.MessageUtil.create(node, {"config": value}, 1)
        service['grpc_client'].open_and_send(msg)

    return {"result": "success"}


@app.get("/job")
def show_job(id: int):
    value = fpm.Value()
    value.sint64 = id
    msg = mu.MessageUtil.create(node, {"id": value}, 2)
    job = service['grpc_client'].open_and_send(msg)
    return {"item_id": id, "title": job.name}


def init_grpc_client(args):
    service['grpc_client'] = GrpcClient("web", args.grpc_host, args.grpc_port)



def parse_args():
    parser = argparse.ArgumentParser("WebServer")
    parser.add_argument('--config', default='./web_config.yml')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    args.grpc_host = config["grpc_server"]["host"]
    args.grpc_port = config["grpc_server"]["port"]
    # args.port = config["port"]
    return args


if __name__ == "__main__":
    # args = parse_args()
    # app.args = args
    # init_grpc_client(args)

    uvicorn.run("main:app", port=5000, log_level="info")


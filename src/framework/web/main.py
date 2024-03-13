import argparse
from fastapi import FastAPI, Form, UploadFile
import uvicorn

from framework.client.grpc_client import GrpcClient
import framework.common.MessageUtil as mu
import framework.protos.node_pb2 as fpn
import framework.protos.message_pb2 as fpm
from typing_extensions import Annotated
import json
from framework.common.yaml_loader import load_yaml
import framework.common.logger_util as logger_util
from contextlib import asynccontextmanager

service = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = parse_args()
    print(args)
    init_grpc_client(args)
    yield
    service.clear()


app = FastAPI(lifespan=lifespan)

node = fpn.Node(node_id="web")

logger = logger_util.get_logger("web_server")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/job/upload")
async def upload_job(file: UploadFile):
    if file is None:
        return {"result": "error", "message": "No file exists"}
    contents = await file.read()
    value = fpm.Value()
    value.string = contents
    msg = mu.MessageUtil.create(node, {"config": value}, 1)
    result = service['grpc_client'].open_and_send(msg)
    job_id = result.named_values['job_id'].sint64
    return {"result": "success", "job_id": job_id}


@app.post("/job")
def create_job(config: Annotated[str, Form()]):
    with open(config, "r") as f:
        data = f.read()
        value = fpm.Value()
        value.string = data
        msg = mu.MessageUtil.create(node, {"config": value}, 1)
        result = service['grpc_client'].open_and_send(msg)
        job_id = result.named_values['job_id'].sint64
        return {"result": "success", "job_id": job_id}


@app.get("/job")
def show_job(id: int):
    value = fpm.Value()
    value.sint64 = id
    msg = mu.MessageUtil.create(node, {"id": value}, 2)
    result = service['grpc_client'].open_and_send(msg)
    job_str = result.named_values['job'].string
    job = json.loads(job_str)
    return job


@app.get("/messages")
def show_message():
    return []


@app.post("/message")
async def send_message(msg: Annotated[str, Form()], file: UploadFile):
    if file is None:
        return {"result": "error", "message": "No file exists"}
    contents = await file.read()
    value = fpm.Value()
    value.string = contents
    msg_value = fpm.Value()
    msg_value.string = msg
    msg = mu.MessageUtil.create(node, {"config": value, "message": msg_value}, 1)
    result = service['grpc_client'].open_and_send(msg)
    job_id = result.named_values['job_id'].sint64
    return {"result": "success", "job_id": job_id}


def init_grpc_client(args):
    service['grpc_client'] = GrpcClient("web", -1, args.grpc_host, args.grpc_port)


def parse_args():
    parser = argparse.ArgumentParser("WebServer")
    parser.add_argument('--config', default='./web_config.yml')
    args = parser.parse_args()
    config = load_yaml(args.config)
    args.grpc_host = config["grpc_server"]["host"]
    args.grpc_port = config["grpc_server"]["port"]
    return args


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info")

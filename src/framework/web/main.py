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
from argparse import Namespace
import os

service = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = parse_args()
    print(args)
    init_grpc_client(args)
    yield
    service['grpc_client'].close()
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
    msg = Namespace()
    msg.data = {"config": contents}
    msg.type = fpm.CREATE_JOB
    result = service['grpc_client'].parse_message(msg)
    job_id = result['job_id']
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
    msg = Namespace()
    msg.data = id
    msg.type = fpm.QUERY_JOB
    job = service['grpc_client'].parse_message(msg)
    return job


@app.get("/start")
def start_model(model_id: str):
    msg = Namespace()
    msg.data = model_id
    msg.type = fpm.LOAD_MODEL
    service['grpc_client'].parse_message(msg)
    return {"result": "success"}


@app.get("/messages")
def show_message():
    return []


@app.post("/message")
async def send_message(msg: Annotated[str, Form()]):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": msg}
    ]
    msg = Namespace()
    msg.data = {"config": service['contents'], "messages": messages}
    msg.type = fpm.CREATE_JOB
    result = service['grpc_client'].parse_message(msg)
    job_id = result['job_id']
    return {"result": "success", "job_id": job_id}


def init_grpc_client(args):
    service['grpc_client'] = GrpcClient("web", 0, args.grpc_host, args.grpc_port, args.compression)
    service['contents'] = read_json_config()


def parse_args():
    parser = argparse.ArgumentParser("WebServer")
    parser.add_argument('--config', default='./web_config.yml')
    args = parser.parse_args()
    config = load_yaml(args.config)
    args.grpc_host = config["grpc_server"]["host"]
    args.grpc_port = config["grpc_server"]["port"]
    args.compression = config["grpc_server"]["compression"]
    return args


def read_json_config():
    config_path = os.path.join(os.path.dirname(__file__), "../../configs/llm_configs/dev_llm_inference.json")
    with open(config_path, "r") as f:
        contents = f.read()
    return contents


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info")

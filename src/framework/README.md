# Distributed VFLAIR

## Introduction

VFLAIR supports training and evaluation after splitting large models. There are two modes for training and evaluation: local mode and distributed mode. This article introduces the technical architecture and usage of distributed mode.

## Distributed framework code structure

```
VFLAIR
├── src
│   ├── framework
│   | ├── client # grpc client and passive party related functions
│ │   | ├── ... #
│   | ├── common # common modules
│ │   | ├── logger_util # Log
│ │   | ├── ... #
│   | ├── credentials # SSL certificates
│ │   │   ├── ...
│   | ├── database # database for saving job and task information
│ │   │   ├── ...
│   | ├── protos # grpc protocols
│ │   │   ├── ...
│   | ├── server # grpc server and active party related functions
│ │   │   ├── ...
│   | ├── README.md # Guidance for distributed VFLAIR
│   | ├── docs # documents
│   | ├── web
├── requirements.txt # installation requirement, we mainly use pytorch3.8 for experiments
```

## Get started

### Build a docker image
1. Download the source code
2. Install the docker environment, see [Docker installation](https://docs.docker.com/engine/install/)
3. Go to the directory and use the following command to build a docker image
- docker build -f Dockerfile-server -t vflair-server .
- docker build -f Dockerfile-web -t vflair-web .

### Installation and deployment

This system currently uses docker to run services. The typical deployment architecture is two machines, one of which runs the Active Party end, and the other runs the Passive Party and Web end
Assume that the server end ip is 192.168.10.3, the passive party(web) end is 192.168.10.5, and the database address is 192.168.10.5

- Run the Active Party side (Server): docker run --name server -d --gpus all -p3333:3333 -v
/home/shan/dev/tools/nlp:/home/shan/dev/tools/nlp vflair-server
- Run the Passive Party side (Web): docker run --name web --add-host=vflair-server.com:192.168.10.3 -e MYSQL_HOST=192.168.10.5 -d -p5000:5000 -e
GRPC_SERVER=vflair-server.com vflair-web

#### Parameter description
1. Active Party side parameters:
- -p3333:3333 indicates that the GRPC port is 3333
- -v Map local model and dataset directories to containers

2. Passive Party side parameter description:
- --add-host Add host mapping. Since VFLAIR uses an SSL certificate, the default certificate domain name is vflair-server.com. It's necessary to modify /etc/hosts and add local resolution 192.168.10.3 vflair-server.com
- MYSQL_HOST: database server host,
- -p5000:5000 opens port 5000,
- GRPC_SERVER=vflair-server.com means the GRPC server address to be connected

More parameter descriptions see [reference](./docs/README_parameters.md)

#### GPU Requirements
The GPU and other information about Active Party and Passive Party we used to test is as follows:

| model                    | dataset  | Active Party Info                                                  | Passive Party Info                                             |
|--------------------------|----------|--------------------------------------------------------------------|----------------------------------------------------------------|
| Qwen2-72B                | huanhuan | GPU: A100-80G *4  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: A100-80G *1<br/> Memory: 245GB<br/>CPU: Xeon(R) 8336C *14 |
| llama-2-7b               | Lambada  | GPU: A100-80G *1  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: A100-80G *1<br/> Memory: 245GB<br/>CPU: Xeon(R) 8336C *14 |
| chatglm3-6b              | Lambada  | GPU: A100-80G *1  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: 4080 *1<br/> Memory: 32GB<br/>CPU:  i7-14700K *28         |
| gemma-2b                 | Lambada  | GPU: A100-80G *1  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: 4080 *1<br/> Memory: 32GB<br/>CPU:  i7-14700K *28         |
| Baichuan-7B              | Lambada  | GPU: A100-80G *1  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: A100-80G *1<br/> Memory: 245GB<br/>CPU: Xeon(R) 8336C *14 |
| Mistral-7B-Instruct-v0.2 | Lambada  | GPU: A100-80G *1  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: A100-80G *1<br/> Memory: 245GB<br/>CPU: Xeon(R) 8336C *14 |



### How to use

Modify the configuration file, assuming it is saved in the basic_config_cola.json file,

1. Initiate a task: use curl localhost:5000/job/upload -F "file=@/Users/test/Downloads/basic_config_cola.json" -v

curl is a tool for sending http requests

Interface API: localhost:5000/job/upload

-F parameter passes the configuration file path

2. After the command runs successfully, the task id will be returned. We can view the task result based on the id.
3. View the task result: Use curl localhost:5000/job?id=1

For more configuration files, refer to [Configuration File Directory](../configs/test_configs)

### Custom Algorithm Steps
Refer to [Add New Algorithms](../../usage_guidance/Add_New_Algorithm.md) After implementing the new algorithm, use the docker build command to build the images separately, and then install and recreate the container.

- docker build -f Dockerfile-server -t vflair-server .
- docker build -f Dockerfile-web -t vflair-web .

## More Documents

- [Distributed Architecture](docs/README_architecture.md)
- [Parameter Configuration](../configs/README.md)
- [Introduction to VFLAIR Federated Learning](../../README.md)
- [VFLAIR LLM Introduction](../configs/README_LLM.md)
- [Docs in Chinese](./README_zh.md)
- [Known issues](./docs/README_issues.md)
- [How to use on Mobile](./docs/README_mobile.md)
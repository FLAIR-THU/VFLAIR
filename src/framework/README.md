# 分布式 VFLAIR

## 介绍
VFLAIR支持对大模型切分后进行训练和评估。训练和评估的方式有本地模式和分布式模式，本文介绍分布式下的技术架构及使用方法。

## 分布式框架代码结构

```
VFLAIR
├── src
│   ├── framework           
│   |   ├── client                      # grpc client and passive party related functions
│   │   |   ├── ...                    # 
│   |   ├── common                      # common modules
│   │   |   ├── logger_util            # Log
│   │   |   ├── ...                    #    
│   |   ├── credentials                 # SSL certificates
│   │   │   ├── ...   
│   |   ├── database                    # database for saving job and task information
│   │   │   ├── ...   
│   |   ├── protos                      # grpc protocols
│   │   │   ├── ...   
│   |   ├── server                       # grpc server and active party related functions 
│   │   │   ├── ...   
│   |   ├── README.md                  # Guidance for distributed VFLAIR 
│   |   ├── docs                        # documents
│   |   ├── web
├── requirements.txt                     # installation requirement, we mainly use pytorch3.8 for experiments
```


## 开始

### 安装部署
本系统目前使用docker方式运行服务，典型的部署架构是两台机器，其中一台运行Active Party端，另一台台运行Passive Party及Web端

- 运行server端：docker run --name server -d --gpus all -e MYSQL_HOST=192.168.10.3 -p3333:3333 -v /home/shannon/dev/tools/nlp:/home/shannon/dev/tools/nlp vflair-server
- 运行client端：docker run --name client -d --gpus all -e GRPC_SERVER=192.168.10.3 -v /home/shannon/dev/tools/nlp:/home/shannon/dev/tools/nlp vflair-client
- 运行Web端：docker run --name web --add-host=vflair-server.com:192.168.10.3 -d -p5000:5000 -e GRPC_SERVER=vflair-server.com vflair-web


### 开始使用

修改配置文件，假设保存在basic_config_cola.json文件中，
1. 发起任务：使用curl localhost:5000/job/upload -F "file=@/Users/test/Downloads/basic_config_cola.json" -v
2. 命令运行成功后会返回任务的id, 我们可以根据id，查看任务的结果
3. 查看任务结果：使用 curl localhost:5000/job?id=1

### 自定义算法步骤
修改源码后，使用docker build命令分别构建映像，然后安装重新创建容器使用即可
- docker build -f Dockerfile-client -t vflair-client .
- docker build -f Dockerfile-server -t vflair-server .
- docker build -f Dockerfile-web -t vflair-web .

## 更多文档
- [分布式架构](docs/README_architecture.md)
- [参数配置]()
- [VFLAIR联邦学习介绍](../../README.md)
- [VFLAIR LLM介绍](../configs/README_LLM.md)
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

### 一、构建docker映像
1. 下载源代码
2. 安装docker环境，参见[Docker安装](https://docs.docker.com/engine/install/)
3. 进行目录，使用如下命令构建docker映像
- docker build -f Dockerfile-server -t vflair-server .
- docker build -f Dockerfile-web -t vflair-web .

### 二、安装部署

本系统目前使用docker方式运行服务，典型的部署架构是两台机器，其中一台运行Active Party端，另一台台运行Passive Party及Web端
假设server端ip是192.168.10.3，passive party及web的ip地址是192.168.10.5，数据库地址是192.168.10.5

- 运行Active Party端(Server)：docker run --name server -d --gpus all -p3333:3333 -v
  /home/shan/dev/tools/nlp:/home/shan/dev/tools/nlp vflair-server
- 运行Passive Party端(Web)：docker run --name web --add-host=vflair-server.com:192.168.10.3 -e MYSQL_HOST=192.168.10.5 -d -p5000:5000 -e
  GRPC_SERVER=vflair-server.com vflair-web

#### 参数说明
1. Active Party端参数：
- -p3333:3333 表明GRPC端口是3333 
- -v 映射本地模型及数据集目录至容器

2.Passive Party端参数说明：
- --add-host 增加host映射。 由于VFLAIR使用了SSL证书，默认证书域名是修改/etc/hosts，加入本地解析 192.168.10.3 vflair-server.com 
- MYSQL_HOST:数据库环境变量， 
- -p5000:5000开启5000端口， 
- GRPC_SERVER=vflair-server.com是需要连接的GRPC服务器地址 

更多参数说明[参考](./docs/README_parameters.md)

#### 显卡要求
我们测试使用的机器信息如下：

| model                    | dataset  | Active Party Info                                                  | Passive Party Info                                             |
|--------------------------|----------|--------------------------------------------------------------------|----------------------------------------------------------------|
| Qwen2-72B                | huanhuan | GPU: A100-80G *4  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: A100-80G *1<br/> Memory: 245GB<br/>CPU: Xeon(R) 8336C *14 |
| llama-2-7b               | Lambada  | GPU: A100-80G *1  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: A100-80G *1<br/> Memory: 245GB<br/>CPU: Xeon(R) 8336C *14 |
| chatglm3-6b              | Lambada  | GPU: A100-80G *1  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: 4080 *1<br/> Memory: 32GB<br/>CPU:  i7-14700K *28         |
| gemma-2b                 | Lambada  | GPU: A100-80G *1  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: 4080 *1<br/> Memory: 32GB<br/>CPU:  i7-14700K *28         |
| Baichuan-7B              | Lambada  | GPU: A100-80G *1  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: A100-80G *1<br/> Memory: 245GB<br/>CPU: Xeon(R) 8336C *14 |
| Mistral-7B-Instruct-v0.2 | Lambada  | GPU: A100-80G *1  <br/>Memory: 980GB <br/>CPU: Xeon(R) 8336C *56   | GPU: A100-80G *1<br/> Memory: 245GB<br/>CPU: Xeon(R) 8336C *14 |


### 三、开始使用

修改配置文件，假设保存在basic_config_cola.json文件中，

1. 发起任务：使用curl localhost:5000/job/upload -F "file=@/Users/test/Downloads/basic_config_cola.json" -v

   其中curl是发http请求的工具

   接口地址：localhost:5000/job/upload

   -F参数传递配置文件路径

2. 命令运行成功后会返回任务的id, 我们可以根据id，查看任务的结果
3. 查看任务结果：使用 curl localhost:5000/job?id=1

更多配置文件参考[配置文件目录](../configs/test_configs)

#### 自定义算法步骤
参考[Add New Algorithms](../../usage_guidance/Add_New_Algorithm.md)实现新算法后，使用docker build命令分别构建映像，然后安装重新创建容器使用即可

- docker build -f Dockerfile-server -t vflair-server .
- docker build -f Dockerfile-web -t vflair-web .

## 更多文档

- [分布式架构](docs/README_architecture.md)
- [参数配置](../configs/README.md)
- [VFLAIR联邦学习介绍](../../README.md)
- [VFLAIR LLM介绍](../configs/README_LLM.md)
- [手机端使用方案](./docs/README_mobile.md)
- [已知问题](./docs/README_issues.md)
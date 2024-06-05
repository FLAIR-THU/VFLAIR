### How to generate python codes.
1. 如果未安装，需要先安装grpc tools 
```shell
pip install grpcio-tools
```

2. 以生成service proto python代码为例，进入src目录，运行如下命令 
```shell
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. ./framework/protos/service.proto
```

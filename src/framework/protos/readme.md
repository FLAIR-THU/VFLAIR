Generate python codes.
pip install grpcio-tools
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. ./framework/protos/service.proto
## GRPC Server parameter description
1. -e CUDA_VISIBLE_DEVICES: Set the GPU that CUDA can use, default is 0
2. -e GRPC_PORT: Set the GRPC listening port, default is 3333
3. -e COMPRESSION: Set the GRPC communication compression protocol, default is None, optional value is GZIP, indicating that GZIP compression is enabled
4. -v Set the host container directory mapping

## GRPC Client parameter description
1. -e CUDA_VISIBLE_DEVICES: Set the GPU that CUDA can use, default is 0
2. -e GRPC_PORT: Set the GRPC port, default is 3333
3. -e GRPC_SERVER: Set the GRPC server address, default is localhost
4. -e COMPRESSION: Set the GRPC communication compression protocol, default is None, optional value is GZIP, indicating that GZIP compression is enabled
5. -e MYSQL_HOST: Set the MySQL server address, default is localhost
6. -e MYSQL_PASSWORD: Set the MYSQL server password, the default is abcd1234
7. -e MYSQL_DB: Set the database name to be used, the default is vflair
8. -e MYSQL_USER: Set the database user to be used, the default is root
9. -e MYSQL_PORT: Set the database port number, the default is 3316
10. -v Set the host container directory mapping
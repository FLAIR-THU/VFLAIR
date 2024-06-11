## GRPC Server参数说明
1. -e CUDA_VISIBLE_DEVICES： 设置CUDA可以使用的gpu，默认0
2. -e GRPC_PORT: 设置GRPC监听端口，默认3333
3. -e COMPRESSION： 设置GRPC通信压缩协议，默认None，可选值GZIP，表示启用GZIP压缩
4. -v 设置主机容器目录映射

## GRPC Client参数说明
1. -e CUDA_VISIBLE_DEVICES： 设置CUDA可以使用的gpu，默认0
2. -e GRPC_PORT: 设置GRPC端口，默认3333 
3. -e GRPC_SERVER: 设置GRPC服务器地址，默认localhost 
4. -e COMPRESSION： 设置GRPC通信压缩协议，默认None，可选值GZIP，表示启用GZIP压缩
5. -e MYSQL_HOST: 设置MySQL服务器地址，默认localhost
6. -e MYSQL_PASSWORD: 设置MYSQL服务器端口，默认abcd1234
7. -e MYSQL_DB：设置使用的数据库名，默认vflair
8. -e MYSQL_USER: 设置使用的数据库名，默认root
9. -e MYSQL_PORT: 设置数据库端口号，默认3316 
10. -v 设置主机容器目录映射
# 启动 Elasticsearch 容器
local_path="/mnt/sharedata/ssd_large/users/wsy/project/agent/wiki/data/index"
# sudo chown -R 1000:1000 $local_path
docker run \
    --name es-wiki \
    -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -v ${local_path}:/usr/share/elasticsearch/data \
    -d elasticsearch:8.19.5
    
# 验证启动

# 停止 Elasticsearch 容器
# docker stop es-wiki

# 删除 Elasticsearch 容器
# docker rm es-wiki
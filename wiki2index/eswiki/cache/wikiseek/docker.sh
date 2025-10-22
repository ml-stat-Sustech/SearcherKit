# docker run --rm -it \
#   -p 6000:8080 \
#   -v /mnt/sharedata/ssd_large/users/wsy/project/agent/wiki/dumps:/dumps xanderstrike/wikiseek \
#   -file /dumps/enwiki-20251001-pages-articles-multistream.xml.bz2 \
#   -index /dumps/enwiki-20251001-pages-articles-multistream-index.txt.bz2

docker run \
    -p 8001:8080 \
    -v ./dumps:/dumps xanderstrike/wikiseek \
    -file /dumps/enwiki-20251001-pages-articles-multistream.xml.bz2 \
    -index /dumps/enwiki-20251001-pages-articles-multistream-index.txt.bz2

# curl "http://192.168.77.12:8001/search?q=hello"
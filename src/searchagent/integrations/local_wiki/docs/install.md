```bash
conda create -n searchagent python=3.10
conda activate searchagent
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

```bash
bash scripts/docker_luncher.sh
curl http://192.168.77.12:9200 # 验证
```


```bash
bash scripts/wiki2index.sh 4
```
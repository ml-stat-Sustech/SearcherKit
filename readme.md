### Step 1: 测试Elasticsearch能否使用
```bash
curl http://192.168.77.12:9200 # 验证Elasticsearch
```

### Step 2: 测试tools是否可用
```bash
bash scripts/tool_test.sh
```

### Step 3: 复制retrievers和tools进行调用

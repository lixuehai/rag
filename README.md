# rag

构建本地知识库，通过RAG向量检索示例

# requirements

```shell
pip install fastapi uvicorn langchain addict datasets==2.16.0 oss2 pypdf rapidocr-onnxruntime modelscope torch simplejson sortedcontainers transformers faiss-cpu tiktoken -i https://mirrors.aliyun.com/pypi/simple/


# 用法

## 运行loader.py，解析pdf生成向量库

```
python loader.py
```

## uvicorn构建Web服务

```shell
uvicorn retrieval:app --reload
```

## 访问

```
http://127.0.0.1:8000
```
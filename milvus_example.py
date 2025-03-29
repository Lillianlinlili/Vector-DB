from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    # 连接到 Milvus 服务器
    # 注意：这里假设 Milvus 服务器运行在本地，端口为 19530
    connections.connect(host='localhost', port='19530')

    # 定义集合的字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # MiniLM-L6 输出维度是 384
    ]

    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="文本向量集合")

    # 创建集合
    collection_name = "text_collection"
    if collection_name in utility.list_collections():
        utility.drop_collection(collection_name)
    collection = Collection(name=collection_name, schema=schema)

    # 加载文本嵌入模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 准备示例数据
    documents = [
        "Milvus 是一个向量数据库",
        "向量数据库用于存储和检索向量数据",
        "Milvus 支持多种向量相似度计算方法"
    ]

    # 生成文本嵌入
    embeddings = model.encode(documents)

    # 准备插入数据
    entities = [{
        "text": text,
        "embedding": embedding.tolist()
    } for text, embedding in zip(documents, embeddings)]

    # 插入数据
    collection.insert(entities)

    # 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    # 加载集合到内存
    collection.load()

    # 执行相似度搜索
    query = "什么是向量数据库？"
    query_embedding = model.encode([query])[0]
    
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    
    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=2,
        output_fields=["text"]
    )

    print("\n查询结果：")
    for hits in results:
        for hit in hits:
            print(f"\n相似度: {1 - hit.distance:.4f}")
            print(f"内容: {hit.entity.get('text')}")

    # 断开连接
    connections.disconnect("default")

if __name__ == "__main__":
    main() 
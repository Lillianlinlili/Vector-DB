import chromadb
from chromadb.config import Settings

def main():
    # 初始化 ChromaDB 客户端
    client = chromadb.Client(Settings(
        persist_directory="./.chroma"  # 数据持久化目录
    ))

    # 创建或获取集合
    collection = client.create_collection(
        name="my_collection",
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )

    # 准备一些示例数据
    documents = [
        "ChromaDB 是一个向量数据库",
        "向量数据库用于存储和检索向量数据",
        "ChromaDB 支持多种向量相似度计算方法"
    ]
    
    # 生成一些示例 ID
    ids = [f"doc_{i}" for i in range(len(documents))]

    # 添加文档到集合
    collection.add(
        documents=documents,
        ids=ids
    )

    # 执行相似度搜索
    query = "什么是向量数据库？"
    results = collection.query(
        query_texts=[query],
        n_results=2  # 返回最相似的 2 个结果
    )

    print("\n查询结果：")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"\n{i+1}. 相似度: {1 - distance:.4f}")
        print(f"   内容: {doc}")

if __name__ == "__main__":
    main() 
import logging
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import pandas as pd
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Establish Milvus connection
connections.connect(host='127.0.0.1', port='19530')

def create_milvus_collection(collection_name, dim):
    logging.info(f"Creating Milvus collection: {collection_name}")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description='search text')
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type': "COSINE",
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='embedding', index_params=index_params)
    logging.info(f"Milvus collection {collection_name} created successfully")
    return collection

def insert_documents_to_collection(collection_name, texts_list, embeddings):
    logging.info(f"Inserting documents into Milvus collection: {collection_name}")
    df = pd.DataFrame({'text':texts_list[:10],'embedding':embeddings})
    df.reset_index(inplace=True)
    df.rename(columns ={"index":"id"}, inplace=True)
    collection = Collection(collection_name)
    collection.insert(df)
    collection.flush()
    num_entities = collection.num_entities
    logging.info(f"{num_entities} documents inserted into Milvus collection {collection_name}")
    return num_entities

def search_documents_in_collection(collection_name, embedding, search_params, output_fields=[], limit=3):
    logging.info(f"Searching documents in Milvus collection: {collection_name}")
    collection = Collection(collection_name)
    collection.load()
    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        output_fields=output_fields,
        limit=limit,
        param=search_params,
    )
    logging.info(f"Search in Milvus collection {collection_name} completed")
    return results

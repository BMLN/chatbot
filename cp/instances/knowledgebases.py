from cp.interfaces.chatbot import Chatbot, batchable

#import chromadb #DEPR
import weaviate

from collections.abc import Iterable
from typing import override


def is_batch(x):
    if len(x) == False:
        return False
    
    dim = None
    for _x in x:
        if isinstance(_x, Iterable) == False:
            return False
        if len(_x) != dim:
            if dim != None:
                return False
            dim = len(_x)

    return True






#DEPR
class ChromaKB(Chatbot.KnowledgeBase):
    def __init__(self, host, port, collection):
        super().__init__()
        self.__client = chromadb.HttpClient(host, port)
        self.__collection = self.__client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    @override
    @batchable(inherent=True)
    def create(self, key, value, id=None):
        self.__collection.add(
            embeddings=key,
            documents=value,
            ids= id if id else (str(key) if is_batch(key) == False else [str(_key) for _key in key])
        )

    @override
    @batchable(inherent=True)
    def retrieve(self, x):
        return self.__collection.get(x)
    
    @override
    @batchable(inherent=True)
    def query(self, **args):
        return self.__collection.query(**args)
    
    @override
    @batchable(inherent=True)
    def search(self, x, **args):
        __args = {
            "query_embeddings": x,
            "n_results": 5,
            "include": ["documents", "distances", "embeddings"]
        }
        __args = __args | {key: value for key, value in args.getitems() if key in __args }

        output = self.__collection.query(**args)
        output["ids"] = output["ids"][0]
        output["distances"] = output["distances"][0]
        output["documents"] = output["documents"][0]

        return output
    



class WeaviateKB(Chatbot.KnowledgeBase):
    
    def __init__(self, host, port, collection):
        super().__init__()
        self.con_cfg = weaviate.connect.ConnectionParams.from_params(
            http_host=host,
            http_port=port,
            http_secure=False,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=False
        )
        self.client = weaviate.WeaviateClient(self.con_cfg, skip_init_checks=True)
        self.client.close()
        with self.client as conn:
            if self.client.collections.exists(str(collection)): #TODO: testing only
                pass
                #self.client.collections.delete(str(collection))

            if self.client.collections.exists(str(collection)) == False:
                self.client.collections.create(
                    name=str(collection),
                    #vectorizer=None, #invalid param, no doc about correct no vectorizer param
                    vector_index_config=weaviate.classes.config.Configure.VectorIndex.hnsw( # or "flat" or "dynamic"
                        distance_metric=weaviate.classes.config.VectorDistances.DOT, # or "cosine"
                        quantizer=None
                    ),
                    properties=[
                        weaviate.classes.config.Property(name="value", data_type=weaviate.classes.config.DataType.TEXT)
                    ]
                )
            self.collection = self.client.collections.get(str(collection))


    @override
    @batchable(inherent=True)
    def create(self, key, value, id=None, batch_size=100):
        
        if is_batch(key) == False:
            keys = [ key ]
            values = [ value ]
            ids = [ id if id else weaviate.util.generate_uuid5(value) ]
        else:
            keys = key
            values = value
            ids = id if id else [ weaviate.util.generate_uuid5(x) for x in values ]

        with self.client as con:
            with self.collection.batch.fixed_size(batch_size=batch_size) as batch:
                for k, v, i in zip(keys, values, ids):
                    batch.add_object(
                        properties= {"value": v},
                        vector= k,
                        uuid = i
                    )

    @override
    @batchable(inherent=True)
    def retrieve(self, x):
        return self.__collection.get(x)
    
    @override
    @batchable(inherent=True)
    def query(self, **args):
        return self.__collection.query(args)
    
    @override
    @batchable(inherent=True)
    def search(self, x, **args):
        #TODO: args + kwargs
        #TODO: fully list/support (all/useful?) of the (potential) args
        __args = {
            "near_vector": x,
            "limit": 5,
            "return_metadata": weaviate.classes.query.MetadataQuery(distance=True)
        }
        __args = __args | {key: value for key, value in args.items() if key in __args }

        with self.client as conn:
            output = self.collection.query.near_vector(**args)

            return output
        
    @override
    @batchable(inherent=True)
    def update(self, x):
        pass

    @override
    @batchable(inherent=True)
    def delete(self, x):
        self.__collection.delete(x)

    def get_client(self):
        return self.client
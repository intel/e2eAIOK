import os.path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Iterable, Any, cast

from pyspark.sql import SparkSession, DataFrame
from ray.data import Dataset

from pyrecdp.core.import_utils import check_availability_and_install, import_sentence_transformers
from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS


class TextEmbedding:
    def __init__(self, embeddings, text_column: str = 'text', embeddings_column: str = 'embedding'):
        # Specify "cuda" to move the model to GPU.
        if embeddings is None:
            raise ValueError(f"embeddings is required")

        from langchain.schema.embeddings import Embeddings
        if not isinstance(embeddings, Embeddings):
            raise ValueError(f"embeddings must a valid implementation of langchain embeddings")

        self.embeddings: Embeddings = embeddings
        self.text_column = text_column
        self.embeddings_column = embeddings_column

    def __call__(self, batch):
        embeddings = self.embeddings.embed_documents(batch[self.text_column])
        return {self.text_column: batch[self.text_column], self.embeddings_column: embeddings}


class DocumentStore(ABC):
    """interface for vector store"""

    def __init__(self, text_column: str,
                 embeddings_column: Optional[str] = 'embedding',
                 embeddings: Optional[str] = 'HuggingFaceEmbeddings',
                 embeddings_args: Optional[Dict] = None,
                 vector_store_args: Optional[Dict] = None,
                 override: bool = False):
        self.text_column = text_column
        self.embeddings_column = embeddings_column
        self.vector_store_args = vector_store_args
        self.override = override
        check_availability_and_install(["langchain"])
        from langchain.schema.embeddings import Embeddings
        self.embeddings: Optional[Embeddings] = self.create_embeddings(embeddings, **embeddings_args)

    def create_embeddings(self, embeddings, **embeddings_args):
        """currently we only use langchain embeddings"""
        if embeddings is None:
            return None

        if embeddings in ['HuggingFaceEmbeddings', 'HuggingFaceInstructEmbeddings', 'HuggingFaceBgeEmbeddings']:
            import_sentence_transformers()
        if 'HuggingFaceInstructEmbeddings' == embeddings:
            check_availability_and_install("InstructorEmbedding")

        from pyrecdp.core.class_utils import new_instance
        embeddings = new_instance('langchain.embeddings', embeddings, **embeddings_args)
        from langchain.schema.embeddings import Embeddings
        assert isinstance(embeddings, Embeddings)
        return embeddings

    def persist(self, ds: Union[Dataset, DataFrame], **kwargs):
        """interface for persist embeddings to underlying vector store"""

        if isinstance(ds, Dataset):
            ds = self.embedding_with_ray(ds, **kwargs)
        else:
            ds = self.embedding_with_spark(ds, **kwargs)

        db = self.do_persist(ds, **kwargs)
        if self.vector_store_args["return_db_handler"]:
            return db
        else:
            return ds

    @abstractmethod
    def do_persist(self, ds: Union[Dataset, DataFrame], **kwargs):
        """base interface for vector store to persist the text and embeddings"""

    def embedding_with_spark(self, df: DataFrame, **kwargs):
        import pandas as pd
        from pyspark.sql import types as T
        def batch_embedding(batches: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            for pdf in batches:
                pdf[self.embeddings_column] = self.embeddings.embed_documents(pdf[self.text_column])
                yield pdf

        fields = [field for field in df.schema] + [T.StructField(self.embeddings_column, T.ArrayType(T.FloatType()))]
        df = df.mapInPandas(batch_embedding, T.StructType(fields))
        return df

    def embedding_with_ray(self, ds: Dataset,
                           batch_size: Optional[int] = None,
                           num_gpus: Optional[int] = None,
                           num_cpus: Optional[int] = None,
                           compute_min_size: Optional[int] = None,
                           compute_max_size: Optional[int] = None):
        from ray.data import ActorPoolStrategy
        ds = ds.map_batches(
            TextEmbedding,
            # Large batch size to maximize GPU utilization.
            # Too large a batch size may result in GPU running out of memory.
            # If the chunk size is increased, then decrease batch size.
            # If the chunk size is decreased, then increase batch size.
            batch_size=batch_size,  # Large batch size to maximize GPU utilization.
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            compute=ActorPoolStrategy(min_size=compute_min_size, max_size=compute_max_size),
            fn_constructor_kwargs={
                "text_column": self.text_column,
                'embeddings_column': self.embeddings_column,
                'embeddings': self.embeddings
            }
        )
        return ds


class LangchainFAAIS(DocumentStore):
    def do_persist(self, ds, **kwargs):
        check_availability_and_install(["faiss-cpu", "faiss-gpu", "langchain"])

        db = self.vector_store_args["db_handler"]
        index_name = self.vector_store_args.get("index", "index")

        rows = ds.iter_rows() if isinstance(ds, Dataset) else ds.collect()
        text_embeddings = [(row[self.text_column], row[self.embeddings_column]) for row in rows]

        from langchain.vectorstores.faiss import FAISS
        if db is not None:
            db.add_embeddings(text_embeddings)
            return db

        if "output_dir" not in self.vector_store_args:
            raise ValueError(f"You must have `output_dir` option specify for FAAIS vector store")
        faiss_folder_path = self.vector_store_args["output_dir"]
        if not self.override and os.path.exists(os.path.join(faiss_folder_path, index_name + ".faiss")):
            db = FAISS.load_local(faiss_folder_path, self.embeddings, index_name)
            db.add_embeddings(text_embeddings)
        else:
            db = FAISS.from_embeddings(text_embeddings, embedding=self.embeddings)

        db.save_local(faiss_folder_path, index_name)
        return db


class LangchainChroma(DocumentStore):
    def persist(self, ds, **kwargs):
        db = self.do_persist(ds, **kwargs)
        if self.vector_store_args["return_db_handler"]:
            return db
        else:
            return ds

    def do_persist(self, ds, **kwargs):
        check_availability_and_install(["chromadb==0.4.15", "langchain"])
        chroma = self.vector_store_args["db_handler"]

        collection_name = self.vector_store_args.get("collection_name", 'langchain')
        rows = ds.iter_rows() if isinstance(ds, Dataset) else ds.collect()
        texts = [row[self.text_column] for row in rows]

        from langchain.vectorstores.chroma import Chroma
        if chroma is not None:
            chroma.add_texts(texts)
            return chroma
        if "output_dir" not in self.vector_store_args:
            raise ValueError(f"You must have `output_dir` option specify for Chroma vector store")
        persist_directory = self.vector_store_args["output_dir"]
        if not self.override and os.path.exists(persist_directory):
            chroma = Chroma(collection_name=collection_name,
                            persist_directory=persist_directory,
                            embedding_function=self.embeddings)
            chroma.add_texts(texts)
        else:
            chroma = Chroma.from_texts(texts,
                                       collection_name=collection_name,
                                       embedding=self.embeddings,
                                       persist_directory=persist_directory)
        chroma.persist()
        return chroma


class HaystackElasticSearch(DocumentStore):
    def persist(self, ds, **kwargs):
        db = self.do_persist(ds, **kwargs)
        if self.vector_store_args["return_db_handler"]:
            return db
        else:
            return ds

    def do_persist(self, ds, **kwargs):
        check_availability_and_install(["farm-haystack", "farm-haystack[elasticsearch7]"])
        exclude_keys = ['db_handler', 'return_db_handler']
        vector_store_args = dict((k, v) for k, v in self.vector_store_args.items() if k not in exclude_keys)
        if isinstance(ds, Dataset):
            class BatchIndexer:
                def __init__(self, text_column: str, vector_store_args: Optional[Dict[str, Any]]):
                    from haystack.document_stores import ElasticsearchDocumentStore
                    self.text_column = text_column
                    self.vector_store_args = vector_store_args
                    self.elasticsearch = ElasticsearchDocumentStore(
                        **vector_store_args
                    )

                def __call__(self, batch):
                    from haystack import Document as SDocument
                    documents = [SDocument(content=text) for text in batch[self.text_column]]
                    self.elasticsearch.write_documents(documents)

            ds.map_batches(BatchIndexer, fn_constructor_kwargs=vector_store_args)
        else:
            def batch_index_with_var(batch, bv_value):
                from haystack import Document as SDocument
                text_column, vector_store_args = bv_value.value
                from haystack.document_stores import ElasticsearchDocumentStore
                elasticsearch = ElasticsearchDocumentStore(
                    **vector_store_args
                )
                documents = [SDocument(content=row[text_column]) for row in batch]
                elasticsearch.write_documents(documents)

            ds = cast(DataFrame, ds)

            bv = ds.sparkSession.sparkContext.broadcast((self.text_column, self.vector_store_args))
            ds.foreachPartition(lambda p: batch_index_with_var(p, bv))

        # share this document store only when rag retrieval want to use document store created from index stage
        elasticsearch = self.vector_store_args["db_handler"]
        if elasticsearch is None:
            from haystack.document_stores import ElasticsearchDocumentStore
            elasticsearch = ElasticsearchDocumentStore(
                **vector_store_args
            )
        return elasticsearch


class DocumentIngestion(BaseLLMOperation):
    def __init__(self,
                 text_column: str = 'text',
                 rag_framework: Optional[str] = 'langchain',
                 embeddings_column: Optional[str] = 'embedding',
                 embeddings: Optional[str] = 'HuggingFaceEmbeddings',
                 embeddings_args: Optional[dict] = None,
                 vector_store: str = 'FAISS',
                 vector_store_args: Optional[dict] = None,
                 override: bool = False,
                 compute_min_size: Optional[int] = None,
                 compute_max_size: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 num_cpus: Optional[int] = None,
                 num_gpus: Optional[int] = None,
                 return_db_handler=False,
                 db_handler=None,
                 requirements=[]):
        """
          Document ingestion operator.
          Args:
            text_column: The name of the column containing the text data.
            rag_framework: The RAG framework to use. The default is 'langchain'.
            embeddings_column: The name of the column to store the embeddings.
            embeddings: The type of embeddings to use. The default is 'HuggingFaceEmbeddings'.
            embeddings_args: Optional arguments for the embeddings.
            vector_store: The type of vector store to use. The default is 'FAISS'.
            vector_store_args: Optional arguments for the vector store.
            override: Whether to override the existing embeddings and vector store.
            compute_min_size: The minimum size of the document to compute embeddings for.
            compute_max_size: The maximum size of the document to compute embeddings for.
            batch_size: The batch size to use when computing embeddings.
            num_cpus: The number of CPUs to use when computing embeddings.
            num_gpus: The number of GPUs to use when computing embeddings.
            return_db_handler: If false, return dataset; If True, return created vectorDB handler.
            db_handler: Use pre-created db_handler as input.
            """
        if rag_framework is None:
            raise ValueError(f"rag framework is required")

        if rag_framework.lower() not in ['langchain', 'haystack']:
            raise ValueError(f"only 'langchain' or 'haystack' rag framework is supported")

        if text_column is None:
            raise ValueError(f"text column is required")

        if vector_store is None:
            raise ValueError(f"vector store is required")

        settings = {
            'text_column': text_column,
            'rag_framework': rag_framework,

            'embeddings_column': embeddings_column,
            'embeddings': embeddings,
            'embeddings_args': embeddings_args,

            'vector_store': vector_store,
            'vector_store_args': vector_store_args,
            'override': override,

            'compute_min_size': compute_min_size,
            'compute_max_size': compute_max_size,
            'batch_size': batch_size,
            'num_gpus': num_gpus,
            'num_cpus': num_cpus,
            'requirements': requirements,

            'return_db_handler': return_db_handler,
            'db_handler': db_handler,
        }
        requirements = requirements
        super().__init__(settings, requirements)
        self.support_ray = True
        self.support_spark = True
        self.rag_framework = rag_framework.lower()
        self.text_column = text_column
        self.embeddings_column = embeddings_column,
        self.embeddings = embeddings
        self.embeddings_args = embeddings_args or {}

        self.compute_args = {
            'compute_min_size': compute_min_size,
            'compute_max_size': compute_max_size,
            'batch_size': batch_size,
            'num_cpus': num_cpus,
            'num_gpus': num_gpus,
        }
        self.vector_store = vector_store.lower()
        self.vector_store_args = vector_store_args or {}
        self.override = override
        self.embeddings_column = embeddings_column
        self.document_store = self._create_document_store()
        self.vector_store_args['return_db_handler'] = return_db_handler
        self.vector_store_args['db_handler'] = db_handler

    def _create_document_store(self) -> DocumentStore:
        document_store_ctor_args = {
            'text_column': self.text_column,
            'embeddings_column': self.embeddings_column,
            'embeddings': self.embeddings,
            'embeddings_args': self.embeddings_args,
            'vector_store_args': self.vector_store_args,
            'override': self.override,
        }

        if 'langchain' == self.rag_framework:
            if 'faiss' == self.vector_store:
                return LangchainFAAIS(**document_store_ctor_args)
            elif 'chroma' == self.vector_store:
                return LangchainChroma(**document_store_ctor_args)
            else:
                raise NotImplementedError(
                    f"vector store {self.vector_store} is not supported yet paired with langchain!")
        else:
            if 'elasticsearch' == self.vector_store:
                document_store_ctor_args['embeddings'] = None
                return HaystackElasticSearch(**document_store_ctor_args)
            else:
                raise NotImplementedError(
                    f"vector store {self.vector_store} is not supported yet paired with haystack!")

    def process_rayds(self, ds: Dataset = None):
        return self.document_store.persist(ds, **self.compute_args)

    def process_spark(self, spark: SparkSession, df: DataFrame = None):
        return self.document_store.persist(df, **self.compute_args)


LLMOPERATORS.register(DocumentIngestion)

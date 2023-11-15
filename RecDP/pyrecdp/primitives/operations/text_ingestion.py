from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Union, Iterable

from pyspark.sql import SparkSession, DataFrame
from ray.data import Dataset

from pyrecdp.core.class_utils import new_instance
from pyrecdp.core.import_utils import import_langchain, import_faiss, import_sentence_transformers
from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS


class TextEmbeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def underlying_embeddings(self):
        """return the underlying embedding model"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """interface for embed texts"""


class VectorStore(ABC):
    """interface for vector store"""

    def __init__(self, text_column: str,
                 vector_store: str,
                 text_embeddings: TextEmbeddings,
                 embeddings_column: str,
                 vector_store_args: Optional[Dict] = None):
        self.text_column = text_column
        self.embeddings_column = embeddings_column
        self.vector_store = vector_store
        self.text_embeddings = text_embeddings
        self.vector_store_args = vector_store_args

    @abstractmethod
    def persist(self, ds: Union[Dataset, DataFrame]):
        """interface for persist embeddings to underlying vector store"""


class LangchainVectorStore(VectorStore):

    def __persist_to_faiss(self, ds):
        import_faiss()
        text_embeddings: List[Tuple[str, List[float]]] = []
        if isinstance(ds, Dataset):
            for row in ds.iter_rows():
                text_embeddings.append((row[self.text_column], row[self.embeddings_column]))
        else:
            for row in ds.collect():
                text_embeddings.append((row[self.text_column], row[self.embeddings_column]))

        from langchain.vectorstores.faiss import FAISS
        db = FAISS.from_embeddings(text_embeddings, embedding=self.text_embeddings.underlying_embeddings())
        if "output_dir" not in self.vector_store_args:
            raise ValueError(f"You must have `output_dir` option specify for vector store {self.vector_store}")

        index_name = self.vector_store_args.get("index", "index")
        db.save_local(self.vector_store_args["output_dir"], index_name)

    def persist(self, ds):
        import_langchain()
        if self.vector_store == "FAISS":
            self.__persist_to_faiss(ds)
        else:
            raise NotImplementedError(f"persist embeddings to vector store '{self.vector_store}' is not supported yet!")


def get_vector_store(text_embeddings: TextEmbeddings,
                     text_column: str = 'text',
                     embeddings_column: str = 'embedding',
                     vector_store: str = 'FAISS',
                     vector_store_args: Optional[Dict] = None,
                     ):
    return LangchainVectorStore(
        text_column=text_column,
        embeddings_column=embeddings_column,
        text_embeddings=text_embeddings,
        vector_store=vector_store,
        vector_store_args=vector_store_args
    )


class DocEmbedding:
    def __init__(self, embeddings: TextEmbeddings, text_column: str = 'text', embeddings_column: str = 'embedding'):
        # Specify "cuda" to move the model to GPU.
        self.embeddings = embeddings
        self.text_column = text_column
        self.embeddings_column = embeddings_column

    def __call__(self, batch):
        embeddings = self.embeddings.embed_documents(batch[self.text_column])
        return {self.text_column: batch[self.text_column], self.embeddings_column: embeddings}


class BaseDocumentIngestion(BaseLLMOperation, ABC):
    def __init__(self,
                 text_column: str = 'text',
                 embeddings_column: str = 'embedding',
                 compute_min_size: Optional[int] = None,
                 compute_max_size: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 num_cpus: Optional[int] = None,
                 num_gpus: Optional[int] = None,
                 settings: Optional[Dict] = None):
        """
            Base class for document ingestion operations.

            Args:
                text_column: The name of the column containing the text of the documents.
                embeddings_column: The name of the column to store the document embeddings.
                compute_min_size: The minimum number of size to computing the document embeddings(If embedding with Ray).
                compute_max_size: The maximum number of size to computing the document embeddings(If embedding with Ray).
                batch_size: The batch size to use when computing the document embeddings(If embedding with Ray).
                num_cpus: The number of CPUs to use when computing the document embeddings(If embedding with Ray).
                num_gpus: The number of GPUs to use when computing the document embeddings(If embedding with Ray).
            """
        settings = settings or {}
        settings.update({
            'text_column': text_column,
            'compute_min_size': compute_min_size,
            'compute_max_size': compute_max_size,
            'batch_size': batch_size,
            'num_gpus': num_gpus,
            'num_cpus': num_cpus,
        })
        super().__init__(settings)
        self.support_ray = True
        self.support_spark = True
        self.compute_min_size = compute_min_size
        self.compute_max_size = compute_max_size
        self.text_column = text_column
        self.batch_size = batch_size
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.embeddings_column = embeddings_column
        self.text_embeddings = self._get_text_embeddings()
        self.vector_store = self._get_vector_store()

    @abstractmethod
    def _get_vector_store(self) -> VectorStore:
        """base interface to get vectoStore"""

    @abstractmethod
    def _get_text_embeddings(self) -> TextEmbeddings:
        """base interface to get text embeddings"""

    def process_rayds(self, ds: Dataset = None):
        from ray.data import ActorPoolStrategy
        ds = ds.map_batches(
            DocEmbedding,
            # Large batch size to maximize GPU utilization.
            # Too large a batch size may result in GPU running out of memory.
            # If the chunk size is increased, then decrease batch size.
            # If the chunk size is decreased, then increase batch size.
            batch_size=self.batch_size,  # Large batch size to maximize GPU utilization.
            num_gpus=self.num_gpus,
            num_cpus=self.num_cpus,
            compute=ActorPoolStrategy(min_size=self.compute_min_size, max_size=self.compute_max_size),
            fn_constructor_kwargs={
                "text_column": self.text_column,
                'embeddings_column': self.embeddings_column,
                'embeddings': self.text_embeddings
            }
        )
        self.vector_store.persist(ds)
        return ds

    def process_spark(self, spark: SparkSession, df: DataFrame = None):
        import pandas as pd
        from pyspark.sql import types as T

        def batch_embedding(batches: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            for pdf in batches:
                pdf[self.embeddings_column] = self.text_embeddings.embed_documents(pdf[self.text_column])
                yield pdf

        fields = [field for field in df.schema] + [T.StructField(self.embeddings_column, T.ArrayType(T.FloatType()))]
        df = df.mapInPandas(batch_embedding, T.StructType(fields))
        self.vector_store.persist(df)
        return df


class DocumentIngestion(BaseDocumentIngestion, TextEmbeddings):

    def __init__(self,
                 text_column: str = 'text',
                 embeddings_column: str = 'embedding',
                 vector_store: str = 'FAISS',
                 vector_store_args: Optional[dict] = None,
                 embeddings: str = 'HuggingFaceEmbeddings',
                 embeddings_args: Optional[dict] = None,
                 compute_min_size: Optional[int] = None,
                 compute_max_size: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 num_cpus: Optional[int] = None,
                 num_gpus: Optional[int] = None):
        """
            Base class for document ingestion operations.

            Args:
                text_column: The name of the column containing the text of the documents.
                embeddings_column: The name of the column to store the document embeddings.
                vector_store: The vector store database to use for storing the document embeddings.
                vector_store_args: A dictionary of arguments to pass to the vector store constructor.
                embeddings: The class name of langchain embedding under module 'langchain.embeddings' to use for embed documents.
                embeddings_args: A dictionary of arguments to pass to the langchain embedding constructor.
                compute_min_size: The minimum number of size to computing the document embeddings(If embedding with Ray).
                compute_max_size: The maximum number of size to computing the document embeddings(If embedding with Ray).
                batch_size: The batch size to use when computing the document embeddings(If embedding with Ray).
                num_cpus: The number of CPUs to use when computing the document embeddings(If embedding with Ray).
                num_gpus: The number of GPUs to use when computing the document embeddings(If embedding with Ray).
            """

        if vector_store is None:
            raise ValueError(f"vector_store is required!")

        if not isinstance(vector_store, str):
            raise ValueError(f"vector_store must be a name of vector store provided in langchain!")

        if embeddings is None:
            raise ValueError(f"langchain embeddings is required!")

        if not isinstance(embeddings, str):
            raise ValueError(f"embeddings must be a class name of langchain embedding!")

        self.vector_store = vector_store
        self.vector_store_args = vector_store_args or {}
        self.embeddings = embeddings
        self.embeddings_args = embeddings_args or {}

        import_langchain()
        if embeddings == 'HuggingFaceEmbeddings':
            import_sentence_transformers()

        from langchain.schema.embeddings import Embeddings
        self.embeddings_model: Embeddings = new_instance("langchain.embeddings", embeddings, **embeddings_args)
        super().__init__(
            text_column=text_column,
            embeddings_column=embeddings_column,
            compute_min_size=compute_min_size,
            compute_max_size=compute_max_size,
            batch_size=batch_size,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            settings={
                'vector_store': vector_store,
                'vector_store_args': self.vector_store_args,
                'embeddings': embeddings,
                'embeddings_args': self.embeddings_args,
            }
        )
        self.support_ray = True
        self.support_spark = True

    def _get_vector_store(self) -> VectorStore:
        """base interface to get vectoStore"""
        return LangchainVectorStore(
            text_column=self.text_column,
            embeddings_column=self.embeddings_column,
            text_embeddings=self.text_embeddings,
            vector_store=self.vector_store,
            vector_store_args=self.vector_store_args
        )

    def _get_text_embeddings(self) -> TextEmbeddings:
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings_model.embed_documents(texts)

    def underlying_embeddings(self):
        return self.embeddings_model


LLMOPERATORS.register(DocumentIngestion)

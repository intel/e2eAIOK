import init_haystack
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

from haystack.retriever.dense import EmbeddingRetriever
from haystack.utils import print_answers
import time

# give a simple test
from haystack.pipeline import FAQPipeline
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
					index="document",
					embedding_field="question_emb",
					embedding_dim=768,
					excluded_meta_data=["question_emb"])

retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=False)
pipe = FAQPipeline(retriever=retriever)

start = time.time()
prediction = pipe.run(query="Python - how to get current date?", top_k_retriever=3)
print_answers(prediction, details="all")
end = time.time()
print('Took %.3f secs' % (end - start))

print('!!!ALL DONE!!!')
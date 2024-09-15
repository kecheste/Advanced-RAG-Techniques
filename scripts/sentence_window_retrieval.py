import os
from llama_index import (
    SimpleDirectoryReader,
    Document,
    StorageContext,
    load_index_from_storage
)

from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank


from decouple import config

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")


documents = SimpleDirectoryReader(
    input_dir="../dataFiles/"
).load_data(show_progress=True)


document = Document(text="\n\n".join([doc.text for doc in documents]))

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = OpenAIEmbedding()

sentence_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    node_parser=node_parser,
)

if not os.path.exists("./storage"):
    index = VectorStoreIndex.from_documents(
        [document], service_context=sentence_context
    )

    index.storage_context.persist(persist_dir="./storage")
else:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./storage"),
        service_context=sentence_context
    )


postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)

rerank = SentenceTransformerRerank(
    top_n=2, model="BAAI/bge-reranker-base"
)

sentence_window_engine = index.as_query_engine(
    similarity_top_k=5, node_postprocessors=[postproc, rerank]
)

response = sentence_window_engine.query(
    "What did the president say about covid-19?"
)

print(response)
import warnings
warnings.filterwarnings('ignore')

from llama_index import SimpleDirectoryReader
from llama_index.llms import Gemini
from llama_index import Document
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.node_parser import get_leaf_nodes
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, StorageContext
from llama_index import set_global_service_context
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from utils import get_prebuilt_trulens_recorder

llm = Gemini(model="models/gemini-pro", temperature=0.1)
documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()
document = Document(text="\n\n".join([doc.text for doc in documents]))

print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

nodes = node_parser.get_nodes_from_documents([document])

leaf_nodes = get_leaf_nodes(nodes)
print(leaf_nodes[30].text)

nodes_by_id = {node.node_id: node for node in nodes}

parent_node = nodes_by_id[leaf_nodes[30].parent_node.node_id]
print(parent_node.text)

auto_merging_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    node_parser=node_parser,
)

set_global_service_context(auto_merging_context)

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

automerging_index = VectorStoreIndex(
    leaf_nodes, storage_context=storage_context, service_context=auto_merging_context
)

automerging_index.storage_context.persist(persist_dir="./merging_index")

automerging_retriever = automerging_index.as_retriever(
    similarity_top_k=12
)

retriever = AutoMergingRetriever(
    automerging_index.service_context,
    automerging_retriever,
    automerging_index.storage_context,
    verbose=True
)

rerank = SentenceTransformerRerank(top_n=6, model="BAAI/bge-reranker-base")

auto_merging_engine = RetrieverQueryEngine.from_args(
    automerging_retriever, node_postprocessors=[rerank], verbose=True
)

auto_merging_response = auto_merging_engine.query(
    "How to I build a portfolio of AI projects??"
)
print(str(auto_merging_response))

tru_recorder_automerging = get_prebuilt_trulens_recorder(auto_merging_engine,
                                            app_id="Automerging Query Engine")

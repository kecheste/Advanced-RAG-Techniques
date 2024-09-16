import os
import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset

from llama_index.llms.openai import OpenAI
import nest_asyncio
import random
import asyncio
import numpy as np

from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    PairwiseComparisonEvaluator,
)

from llama_index.core.evaluation.eval_utils import (
    get_responses,
    get_results_df,
)
from llama_index.core.evaluation import BatchEvalRunner

from collections import defaultdict
import pandas as pd

os.environ["OPENAI_API_KEY"] = "sk-..."

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.evaluation.eval_utils import (
    get_responses,
    get_results_df,
)
from llama_index.core.evaluation import BatchEvalRunner

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

text_splitter = SentenceSplitter()

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

documents = SimpleDirectoryReader(
    input_files=["./Summary.pdf"]
).load_data()

nodes = node_parser.get_nodes_from_documents(documents)

base_nodes = text_splitter.get_nodes_from_documents(documents)

sentence_index = VectorStoreIndex(nodes)

base_index = VectorStoreIndex(base_nodes)

query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
window_response = query_engine.query(
    "What are the concerns surrounding the AMOC?"
)
print(window_response)

window = window_response.source_nodes[0].node.metadata["window"]
sentence = window_response.source_nodes[0].node.metadata["original_text"]

print(f"Window: {window}")
print("------------------")
print(f"Original Sentence: {sentence}")

for source_node in window_response.source_nodes:
    print(source_node.node.metadata["original_text"])
    print("--------")

nest_asyncio.apply()

len(base_nodes)

num_nodes_eval = 30
sample_eval_nodes = random.sample(base_nodes[:200], num_nodes_eval)

dataset_generator = DatasetGenerator(
    sample_eval_nodes,
    llm=OpenAI(model="gpt-4"),
    show_progress=True,
    num_questions_per_chunk=2,
)

eval_dataset = await dataset_generator.agenerate_dataset_from_nodes()

eval_dataset.save_json("data/ipcc_eval_qr_dataset.json")

eval_dataset = QueryResponseDataset.from_json("data/ipcc_eval_qr_dataset.json")

evaluator_c = CorrectnessEvaluator(llm=OpenAI(model="gpt-4"))
evaluator_s = SemanticSimilarityEvaluator()
evaluator_r = RelevancyEvaluator(llm=OpenAI(model="gpt-4"))
evaluator_f = FaithfulnessEvaluator(llm=OpenAI(model="gpt-4"))

max_samples = 30

eval_qs = eval_dataset.questions
ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

base_query_engine = base_index.as_query_engine(similarity_top_k=2)

query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,

    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

base_pred_responses = get_responses(
    eval_qs[:max_samples], base_query_engine, show_progress=True
)
pred_responses = get_responses(
    eval_qs[:max_samples], query_engine, show_progress=True
)

pred_response_strs = [str(p) for p in pred_responses]
base_pred_response_strs = [str(p) for p in base_pred_responses]

evaluator_dict = {
    "correctness": evaluator_c,
    "faithfulness": evaluator_f,
    "relevancy": evaluator_r,
    "semantic_similarity": evaluator_s,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

eval_results = await batch_runner.aevaluate_responses(
    queries=eval_qs[:max_samples],
    responses=pred_responses[:max_samples],
    reference=ref_response_strs[:max_samples],
)

base_eval_results = await batch_runner.aevaluate_responses(
    queries=eval_qs[:max_samples],
    responses=base_pred_responses[:max_samples],
    reference=ref_response_strs[:max_samples],
)

results_df = get_results_df(
    [eval_results, base_eval_results],
    ["Sentence Window Retriever", "Base Retriever"],
    ["correctness", "relevancy", "faithfulness", "semantic_similarity"],
)
display(results_df)
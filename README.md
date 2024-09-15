# Advanced RAG Techniques

The standard RAG pipeline uses the same text chunk for embedding and synthesis. The issue with this approach is that the embedding-based retrieval works well with smaller chunks whereas LLM needs more context and bigger chunks to synthesize a good answer.

- There are two advanced retrieval techniques we are going to explore in this research.

## Sentence-window Retrieval

In this method, we retrieve based on smaller sentences to better match the relevant context and then synthesize based on the expanded context window around the sentence. We first embed smaller chunks or sentences and store them in a vector database.

We also add the context of sentences that occur before and after each chunk. During retrieval, we retrieve the sentences that are more relevant to the question with the similarity search and then replace them with the full surrounding context. This allows us to expand the context that is being fed to the LLM.

## Auto-Merging Retrieval

Another issue with a naive approach is that you are retrieving a bit of fragmented context chunks to put into the LLM context window. The fragmentation gets worse with a smaller chunk size. For instance, you might get back two or more retrieved-context chunks roughly in the same section, however, there are no guarantees on the ordering of these chunks.

This can potentially impact the LLM’s ability to synthesize the information over this retrieved context within its context window.

- More will be discussed in the _Summary.pdf_ file

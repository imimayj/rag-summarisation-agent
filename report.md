# RAG Summarisation Agent Report

For this exercise, I attempted to build a proof-of-concept (POC) retrieval-augmented generation (RAG) agent to retrieve and summarise publically-available data from Wikipedia pages and evaluate the pipeline utilising known NLP techniques.

## The Approach

I have previously used my own scripts constructed on top of the OpenAI Evals library (using open-source academic datasets which include turn-programs) with API-accessible and Ollama-hosted free models to conduct such a task. However, in this approach, I decided that I would try to implement my own learning of vector generation and retrieval in order to build this RAG agent, to ensure reproducibility. As such, we use the Wikipedia library to pull the information from two pages (Non-disclosure_agreement and History_of_physics), along with FastEmbed's TextEmbedding functionality to embed the pages' chunked data into manageable chunks using jinaai/jina-embeddings-v2-base-en. We then identify the top n closest chunks to the target input questions (\n **"What are the key discoveries in modern physics?"** and \n **"What is the structure of a non-disclosure agreement?"**) based on cosine similarity thresholding. Finally, we feed each of these chunks' embeddings through HuggingFace's bart-large-cnn utilising the transformers summarisation pipeline, outputting a concatenated version of the summary.

## Evaluation

Evaluation is conducted following the generation of the summarisation reports of the 'closest' results from each page (src/evaluation.py). This includes:

* ROUGE: Metrics used to evaluate the quality of summaries by comparing n-grams between the model's generated summary and the reference summary provided (more on that later). The evaluation encompasses ROUGE-1 (overlap of unigrams between generated and reference), ROUGE-2 (overlap of bigrams between generated and reference), and ROUGE-L (longgest common subsequence between generated and reference)
* BLEU: A metric used to evaluate the precision of n-grams present in the generated summary compared to the reference summary. Closer to 1 is better.
* Precision, Recall, and F1-Score: Metrics which measure the true vs false positives (in this case, which words should and shouldn't be included in the generated summary) and the harmonic mean of precision and recall (f1).
* Semantic Similarity: A metric to evaluate the cosine similarity of the generated embeddings when compared to the reference summary.

The evaluation script is run at the end of the pipeline, and outputs a report (eval_report.json) upon completion.

## Challenges and Limitations

- My initial (potentially optimistic) approach comprised only of generating a list of embeddings for all chunks in the page, identifying the top n using cosine similarity, and then feeding this through the summarisation model. This was a challenge as bart-large has a max input sequence length of 1024, and thus much information was being left out - impacting the summaries. This was addressed by summarising each of the top n chunks and then concatenating (and, in the case of the NDA summary, formatting) these summaries.
- The max input sequence length of such models (including embeddings models - jina also only has a max input length of 768 tokens) proved to be a little gnarly. I have previously identified this issue in some of my other work, and the only solution I have identified to be consistently reliable is to chunk texts so that they remain within this maximum length. Admittedly we would likely not face this issue with API-accessible models, though the output is decidedly less transparent.
- The reference summary currently utilised to generate the evaluation report is, at present, only using the first 1000 characters of each of the pages. This is a considerable limitation due to the likelihood that this is not encapsulating the correct information which we are trying to identify, and impacts semantic similarity scores due to the evident linguistic similarities between this and the input text fed to each of the models. With more time, I would perhaps use an API-accessible LLM to generate a summary, or use a reliable human annotator for the task.
- Evaluation metrics: I utilised many of the metrics which I have encountered in NLP tasks. I do believe that such metrics are valuable, though I do think a quantifiable ground truth validation/test dataset would be more valuable for ensuring accuracy of results. 
- FastEmbed, while a thoroughly efficient and impressive tool in my opinion, only supports a variety of 24 different models from providers such as jinaai, snowflake, sentence-transformers, thenlper, BAAI, and nomic AI. This does limit the embeddings models which could be used for the task. However, I do believe that its native integration with efficient vector dbs such as qdrant, as well as its speed, make it an impressive choice for the task.

## Future Work
- Test out different models for generating embeddings, and for summarisation using both open-source libraries and api-accessible models.
- Refine the reference summaries so we can more accurately assess models' performance of the task.
- Further modularise and improve the code, to create a fully agentic implementation utilising CLI args which could (theoretically) point to any page on the internet.
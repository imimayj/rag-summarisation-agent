import wikipedia, torch, json, logging
import numpy as np
from fastembed import TextEmbedding
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

from .evaluation import execute_evaluation

logger = logging.getLogger(__name__)

def chunk_text(text, chunk_size=600):
    """
    Split text into chunks of a given size (to avoid out of input context length-related errors)
    """
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def embed_text_chunks(text_chunks, embedding_model):
    """
    Embeds chunks of text using FastEmbed model
    """
    embeddings = []
    for chunk in text_chunks:
        embedding = list(embedding_model.passage_embed(chunk))
        embeddings.append(embedding[0])

    return np.array(embeddings)

def retrieve_relevant_chunk(query, text_chunks, embeddings, embedding_model, top_n=5, similarity_threshold=0.75):
    """
    Retrieve the top n relevant chunks based on cosine similarity to the query and apply a similarity threshold
    """
    query_embedding = list(embedding_model.query_embed(query))[0]
    #make it 2d 8)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    # compute cosim between query and text chunks
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    #get indices of top n most relevant chunks
    sorted_indices = np.argsort(similarities)[::-1]
    #filter based on threshold
    selected_chunks = []
    selected_similarities = []
    for idx in sorted_indices[:top_n]:
        if similarities[idx] >= similarity_threshold:
            selected_chunks.append(text_chunks[idx])
            selected_similarities.append(similarities[idx])
        else:
            break

    return selected_chunks, selected_similarities

# def summarise_content(chunks, summariser, max_input_length=800):
#     """
#     Concatenate relevant chunks and summarise using HuggingFace's BART model
#     w/ input length management
#       NB THIS PRODUCED PRETTY RUBBISH RESULTS
#     """
#     concatenated_content = ' '.join(chunks)
#     tokenizer = summariser.tokenizer
#     tokens = tokenizer.encode(concatenated_content)
#     if len(tokens) > max_input_length:
#         print("input too long")
#         truncated_tokens = tokens[:max_input_length]
#         concatenated_content = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
#     summary = summariser(concatenated_content, max_length=400, min_length=30, do_sample=False, truncation=True)
#     return summary[0]['summary_text']

def summarise_individual_chunk(chunk, summariser, max_length=150, min_length=30):
    """  
    Summarise a single chunk (to avoid truncation problem with transformers tokenizers)
    """
    summary = summariser(
        chunk,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
        clean_up_tokenization_spaces=True
    )
    return summary[0]['summary_text']

def hierarchical_summarisation(chunks, summariser, task_type="general"):
    """ 
    Hierarchical summarisation: summarise each chunk individually, then combine
    """
    if not chunks:
        return "No relevant info found"

    individual_summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarise_individual_chunk(chunk, summariser)
        if summary:
            individual_summaries.append(summary)
    combined_summary = " ".join(individual_summaries)
    return combined_summary

def format_nda_structure(summary_text):
    """ 
    Convert NDA summary to bullet points
    """
    sentences = summary_text.split('. ')
    bullet_points = []
    for sentence in sentences:
        if sentence.strip():
            clean_sentence = sentence.strip().rstrip('.')
            if clean_sentence:
                bullet_points.append(f"* {clean_sentence}")

    return '\n'.join(bullet_points)

def main():
    history_of_physics = wikipedia.page("History_of_physics").content
    nda = wikipedia.page("Non-disclosure_agreement").content

    physics_chunks = chunk_text(history_of_physics)
    nda_chunks = chunk_text(nda)

    model_name = "jinaai/jina-embeddings-v2-base-en"
    embedding_model = TextEmbedding(model_name=model_name)
    physics_embeds = embed_text_chunks(physics_chunks,embedding_model)
    nda_embeds = embed_text_chunks(nda_chunks,embedding_model)

    query_physics = "What are the key discoveries in modern physics?"
    query_nda = "What is the structure of a non-disclosure agreement?"
    
    relevant_physics_chunks, _ = retrieve_relevant_chunk(query_physics, physics_chunks, physics_embeds, embedding_model, top_n=10)
    logger.info(relevant_physics_chunks)
    relevant_nda_chunks, _ = retrieve_relevant_chunk(query_nda, nda_chunks, nda_embeds, embedding_model)
    logger.info(relevant_nda_chunks)
    
    summariser = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    physics_summary = hierarchical_summarisation(relevant_physics_chunks, summariser)
    nda_summary = hierarchical_summarisation(relevant_nda_chunks, summariser)

    formatted_nda_structure = format_nda_structure(nda_summary)

    logger.info("Summary of Modern Physics Discoveries:")
    logger.info(physics_summary)
    logger.info("\nStructure of a Non-disclosure Agreement:")
    logger.info(formatted_nda_structure)

    reference_summary_physics = wikipedia.page("History_of_physics").content[:1000]
    reference_summary_nda = wikipedia.page("Non-disclosure_agreement").content[:1000]

    evaluation_results = execute_evaluation(reference_summary_physics, physics_summary, reference_summary_nda, formatted_nda_structure)
    evaluation_report = {
        "Physics Summary": {
            "generated_summary": physics_summary,
            "reference_summary": reference_summary_physics,
            "evaluation": evaluation_results["Physics Summary"]
        },
        "NDA Summary": {
            "generated_summary": formatted_nda_structure,
            "reference_summary": reference_summary_nda,
            "evaluation": evaluation_results["NDA Summary"]
        }
    }
    with open("./eval_report.json", 'w') as f:
        json.dump(evaluation_report, f, indent=4)
    
if __name__ == "__main__":
    main()
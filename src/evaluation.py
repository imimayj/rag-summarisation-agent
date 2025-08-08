import wikipedia, logging
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_recall_fscore_support
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

def evaluate_rouge(reference, summary):
    """
    Calculate ROUGE scores: ROUGE-1, ROUGE-2, ROUGE-L
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

def evaluate_bleu(reference, summary):
    """
    Calculate bleu score
    """
    reference_tokens = reference.split()
    summary_tokens = summary.split()
    bleu_score = sentence_bleu([reference_tokens], summary_tokens)
    return bleu_score

def evaluate_prec_rec_f1(reference, summary):
    """
    Calculate Precision, Recall, F1 based on n-grams
    """
    reference_tokens = set(reference.split())
    summary_tokens = set(summary.split())
    
    precision = len(reference_tokens & summary_tokens) / len(summary_tokens) if len(summary_tokens) > 0 else 0
    recall = len(reference_tokens & summary_tokens) / len(reference_tokens) if len(reference_tokens) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate_semantic_similarity(reference, summary, model_name="jinaai/jina-embeddings-v2-base-en"):
    """
    Evaluate the cosine similarity of two text embeddings (semantic similarity)
    """
    embedding_model = TextEmbedding(model_name=model_name)
    reference_embed = list(embedding_model.passage_embed(reference))[0]
    summary_embed = list(embedding_model.passage_embed(summary))[0]

    similarity = np.dot(reference_embed, summary_embed) / (np.linalg.norm(reference_embed)) * np.linalg.norm(summary_embed)
    return similarity

def execute_evaluation(
    reference_summary_physics, 
    generated_summary_physics,
    reference_summary_nda,
    generated_summary_nda
):
    rouge_physics = evaluate_rouge(reference_summary_physics, generated_summary_physics)
    rouge_nda = evaluate_rouge(reference_summary_nda, generated_summary_nda)
    
    # Evaluate on BLEU
    bleu_physics = evaluate_bleu(reference_summary_physics, generated_summary_physics)
    bleu_nda = evaluate_bleu(reference_summary_nda, generated_summary_nda)
    
    # Evaluate Precision, Recall, F1
    precision_physics, recall_physics, f1_physics = evaluate_prec_rec_f1(reference_summary_physics, generated_summary_physics)
    precision_nda, recall_nda, f1_nda = evaluate_prec_rec_f1(reference_summary_nda, generated_summary_nda)
   
    # Evaluate Semantic Cosine Similarity of Embeddings)
    semantic_similarity_physics = evaluate_semantic_similarity(reference_summary_physics, generated_summary_physics)
    semantic_similarity_nda = evaluate_semantic_similarity(reference_summary_nda, generated_summary_nda)

        # Print Results
    logger.info(f"Evaluation Results for Physics Summary:")
    logger.info(f"ROUGE: {rouge_physics}")
    logger.info(f"BLEU: {bleu_physics}")
    logger.info(f"Precision: {precision_physics}, Recall: {recall_physics}, F1: {f1_physics}")
    logger.info(f"Semantic Similarity: {semantic_similarity_physics:.4f}")
    
    logger.info("\nEvaluation Results for NDA Summary:")
    logger.info(f"ROUGE: {rouge_nda}")
    logger.info(f"BLEU: {bleu_nda}")
    logger.info(f"Precision: {precision_nda}, Recall: {recall_nda}, F1: {f1_nda}")
    logger.info(f"Semantic Similarity: {semantic_similarity_nda:.4f}")

    evaluation_results = {
        "Physics Summary": {
            "ROUGE": rouge_physics,
            "BLEU": bleu_physics,
            "Precision": precision_physics,
            "Recall": recall_physics,
            "F1": f1_physics,
            "Semantic Similarity": semantic_similarity_physics
        },
        "NDA Summary": {
            "ROUGE": rouge_nda,
            "BLEU": bleu_nda,
            "Precision": precision_nda,
            "Recall": recall_nda,
            "F1": f1_nda,
            "Semantic Similarity": semantic_similarity_nda
        }
    }

    return evaluation_results
    


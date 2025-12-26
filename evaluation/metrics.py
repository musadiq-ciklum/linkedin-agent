# evaluation/metrics.py

def precision_at_k(retrieved_ids, relevant_ids, k):
    if k == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    return sum(1 for d in retrieved_k if d in relevant_set) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    return sum(1 for d in retrieved_k if d in relevant_set) / len(relevant_set)


def keyword_overlap_score(generated, reference):
    if not generated or not reference:
        return 0.0

    gen_words = set(generated.lower().split())
    ref_words = set(reference.lower().split())

    if not ref_words:
        return 0.0

    return len(gen_words & ref_words) / len(ref_words)

def rp_at_k(gold: list, predicted: list, k: int):
    """
    Calculate Rank Precision at K (RP@K)

    Parameters:
    - gold: List containing the true relevant items
    - predicted: List containing the predicted items in ranked order
    - k: Top K items to consider

    Returns:
    - RP@K (Rank Precision at K) value
    """

    # Ensure k is not greater than the length of the gold list
    gold_k = min(k, len(gold))

    # Retrieve the top K predicted items
    top_k_predicted = predicted[:k]

    # Count the number of true positives in the top K
    true_positives = sum(1 for item in top_k_predicted if item in gold)

    # Calculate RP@K
    rp_at_k = true_positives / gold_k if gold_k > 0 else 0.0

    return rp_at_k


def recall_at_k(gold: list, predicted: list, k: int):
    """
    Calculate Recall at K (Recall@K)

    Parameters:
    - gold: List containing the true relevant items
    - predicted: List containing the predicted items in ranked order
    - k: Top K items to consider

    Returns:
    - Recall@K) (Recall at K) value
    """

    rank = [x in gold for x in predicted]
    recall = sum(rank[:k]) / len(gold)
    return recall


def query_is_formatted_well(example, pred, trace=None):
    """
    Checks if a query is formatted well. A query is formatted well if it is a question and ends with a question mark.
    """
    if pred.query[-1] != "?":
        return False

    if "question" in pred.query.lower():
        return False

    return True

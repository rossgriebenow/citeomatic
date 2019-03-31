from typing import Dict
import pandas as pd
import numpy as np

def gold_citations(doc_id: int, df: pd.DataFrame, min_citations: int, id_dict: Dict):
    cite_ids = df.iloc[doc_id].outCitations
    gold_citations_1 = set(filter(None,[id_dict.get(i) for i in cite_ids]))
    
    if doc_id in gold_citations_1:
        gold_citations_1.remove(doc_id)
        
    citations_of_citations = []
    for c in gold_citations_1:
        cite_ids = df.iloc[c].outCitations
        citations_of_citations.extend(list(filter(None,[id_dict.get(i) for i in cite_ids])))
        
    gold_citations_2 = set(citations_of_citations).union(gold_citations_1)
    
    if doc_id in gold_citations_2:
        gold_citations_2.remove(doc_id)
        
    if len(gold_citations_1) < min_citations:
        return [], []

    return gold_citations_1, gold_citations_2
    
def eval_text_model(instances, reader, embedder, neighbors, ranker, df, id_dict):
    EVAL_KEYS = [1,5,10,20]
    results_1 = []
    results_2 = []
    for instance in instances:
        query_id = instance["query_id"].as_tensor(None).item()
        gold_citations_1, gold_citations_2 = gold_citations(query_id, df, 1, id_dict)
        if len(gold_citations_1) < 10:
	    continue
        
        candidates, scores = neighbors.get_nns_by_instance(instance, embedder)
        
        rank_inst = [reader._instance_from_ids(query_id,c_id, 1.0) for c_id in candidates]
        rankings = ranker.forward_on_instances(rank_inst)
        clean_rankings = [r["cite_prob"][0] for r in rankings]
        order = np.argsort(clean_rankings)
        predictions = np.array(candidates)[order]
        
        r_1 = precision_recall_f1_at_ks(
            gold_y=gold_citations_1,
            predictions=predictions,
            scores=None,
            k_list=EVAL_KEYS
        )

        r_2 = precision_recall_f1_at_ks(
            gold_y=gold_citations_2,
            predictions=predictions,
            scores=None,
            k_list=EVAL_KEYS
        )

        results_1.append(r_1)
        results_2.append(r_2)
        
    averaged_results_1 = average_results(results_1)
    averaged_results_2 = average_results(results_2)
    
    return averaged_results_1, averaged_results_2

def precision_recall_f1_at_ks(gold_y, predictions, scores=None, k_list=None):

    def _mrr(ranked_list):
        try:
            idx = ranked_list.index(True)
            return 1. / (idx + 1)
        except ValueError:
            return 0.0

    if k_list is None:
        k_list = [1, 5, 10]
    if scores is not None:
        sorted_predictions = [p for p, _ in
                              sorted(zip(predictions, scores), key=lambda x : x[1], reverse=True)]
    else:
        sorted_predictions = predictions

    gold_set = set(gold_y)

    sorted_correct = [y_pred in gold_set for y_pred in sorted_predictions]

    results = {
        'precision': [],
        'recall': [],
        'f1': [],
        'mrr': _mrr(sorted_correct),
        'k': k_list
    }
    num_gold = len(gold_y)

    for k in k_list:
        num_correct = np.sum(sorted_correct[:k])
        p = num_correct / k
        r = num_correct / num_gold
        if num_correct == 0:
            f = 0.0
        else:
            f = 2 * p * r / (p + r)
        results['precision'].append(p)
        results['recall'].append(r)
        results['f1'].append(f)

    return results


def average_results(results: list):
    p_matrix = []
    r_matrix = []
    f_matrix = []
    mrr_list = []

    for r in results:
        p_matrix.append(r['precision'])
        r_matrix.append(r['recall'])
        f_matrix.append(r['f1'])
        mrr_list.append(r['mrr'])

    return {
        'precision': list(np.mean(p_matrix, axis=0)),
        'recall': list(np.mean(r_matrix, axis=0)),
        'f1': list(np.mean(f_matrix, axis=0)),
        'mrr': np.mean(mrr_list),
    }


def f1(p, r):
    if p + r == 0.0:
        return 0.0
    else:
        return 2 * p * r / (p + r)

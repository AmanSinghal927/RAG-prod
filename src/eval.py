from fuzzywuzzy import fuzz
import numpy as np

def convert_preds_to_results_array(predictions, correct_answers, threshold=90):
    results = []
    for preds, correct in zip(predictions, correct_answers):
        row = [1 if fuzz.partial_ratio(str(pred).lower(), str(correct).lower()) >= threshold and len(str(pred))>=0.95*len(str(correct)) else 0 for pred in preds]
        results.append(row)
    return np.array(results)

class eval_retrieval:
    def __init__(self):
        pass
    def recall_k(self, test_labels, ret_context, k):
        ctr = 0
        correct, incorrect = [], []
        for i in range(len(ret_context)):
            done = False
            if isinstance(test_labels[i], float) or str(test_labels[i]) == "nan":
                # handle case for when answer is not present in the pdf
                if ret_context[0] == "None":
                    test_labels[i] = "None"
                    ctr += 1
                    done = True
                test_labels[i] = "None"
                
            else:
                for j in range(min(k,len(ret_context[0]))):
                    if fuzz.partial_ratio(test_labels[i], ret_context[i][j])>=90 and len(ret_context[i][j])>=len(test_labels[i])*0.95:
                        ctr += 1
                        done = True
                        break
                    
            if done == False:
                incorrect.append({test_labels[i]:ret_context[i]})

            else:
                correct.append({test_labels[i]:ret_context[i]})

        return ctr/len(ret_context), incorrect, correct
    
    def mean_reciprocal_rank(self, predictions, correct_answers, threshold=90):
        results = convert_preds_to_results_array(predictions, correct_answers, threshold=90)
        reciprocal_ranks = (np.asarray(result).nonzero()[0] for result in results)
        mrr = np.mean([1. / (rank[0] + 1) if rank.size else 0. for rank in reciprocal_ranks]) # if non-zero value is extracted
        return mrr
import heapq
import numpy as np
import pandas as pd
import collections
import itertools
import warnings
import fnmatch

from .model.logit  import LogisticRegression
from .model.order_logit import OrderLogit
from .model.ols import LinearRegression



 
def return_top(scores, varN, models, nbest=100):
    varN = np.array(varN)
    scores = np.array(scores)
    models = np.array(models)

    varN_top = np.array([], np.int32)
    scores_top = np.array([], np.float64)
    models_top = np.array([], np.float64)

    for var in np.unique(varN):
        temp = varN == var
        temp2 = scores[temp].argsort()[-nbest:]
        scores_top = np.append(scores_top, scores[temp][temp2])
        varN_top = np.append(varN_top, varN[temp][temp2])
        models_top = np.append(models_top, models[temp][temp2])

    return scores_top, varN_top, models_top

 



def best_subset_exhaustive(X, y, candidates, weights=None, method='logistic'):
    
    if method.lower() == "ordinal" and weights is not None:
        raise ValueError("Weights are not supported for ordinal regression")
    
    if weights is None:
        weights = np.ones_like(y)

    
 
    candidates = candidates[:]
 

    if weights is None: 
        weights = np.ones_like(y)

    if method not in ["logistic", "ordinal", "ols"]:
        raise ValueError("Method must be one of 'logistic', 'ordinal', or 'ols'")
    
    if method == "logistic" or method is None:
        g, Is, const, candidates = logistic_null_model(y, X, candidates, weights)
    elif method == "ordinal":
        g, Is, const, candidates = ordinal_null_model(y, X, candidates)
    elif method == "ols":   
        g, Is, const, candidates = ols_null_model(y, X, candidates, weights)

 
    
 
    C = candidates[len(const):]
   
    var_nums = list(range(1, len(C) + 1))
    varN = []
    models = []
    scores = []
    count = 0

    for L in var_nums:
        for model in itertools.combinations(C, L):
            temp_model = list(model)
            var = len(model)
            varN.append(var)
 
            temp_model = const + temp_model
            loc = np.nonzero(np.in1d(candidates, temp_model))[0]
            model_string = " ".join(temp_model[len(const):])

            models.append(model_string)

            scoreStat = -g[loc].T.dot(np.linalg.inv(Is[np.ix_(loc, loc)])).dot(g[loc])
            scores.append(scoreStat)
            count += 1

        print(f"Finished Var Family: {L}")

    results_exhaustive = pd.DataFrame({
        "Var Number": varN,
        "Models": models,
        "Scores": scores
    }).sort_values(by=["Var Number", "Scores"], ascending=[True, False])

    print(f"Total Models: {count}")
    return results_exhaustive


def logistic_null_model(y , X, candidates, weights):

    if 'const' not in X.columns.tolist():
        X.insert(0, "const", 1) 
    else:
        X.drop("const", axis=1, inplace=True)
        X.insert(0, "const", 1)

    if "const" not in candidates:
        candidates = ["const"] + candidates
    else:
        candidates.remove("const")
        candidates = ["const"] + candidates
 

    X = np.asarray(X[candidates])
   
    avg = np.sum(weights*y) / np.sum(weights)    
   
    null_model = np.log(avg / (1 - avg))      
   
    theta_0 = np.append(null_model, np.zeros(len(candidates)-1))
 
    g = LogisticRegression.score_function(theta_0, X, y, weights)
    Is = LogisticRegression.information_matrix(theta_0, X, weights)
    const =  ["const"]
    return g, Is, const, candidates
 

def ordinal_null_model(y , X, candidates):

    df = X
    ol = OrderLogit(y, df[[]], 'Fisher', 30, theta_initial=[], descending=True)
    ol.dataprep()
    ol.loss_function()
    ol.fit()
    
    theta_0 =  np.append( ol.theta.copy(),np.zeros(len(candidates)) )
                         
    model = OrderLogit(y, X[candidates], 'Fisher', 30, theta_initial=theta_0, descending=True)
    model.dataprep()
    g =  model.jacobian()
    Is = -model.hessian() # For testing this be turned to # -Is
    model.summary()
    const = list(model.results_summary.index[:np.unique(model.y).shape[0]-1])
    cands_temp = model.columns
    candidates = const + cands_temp

    return g, Is, const, candidates


def ols_null_model(y , X, candidates, weights):

    if 'const' not in X.columns.tolist():
        X.insert(0, "const", 1) 
    else:
        X.drop("const", axis=1, inplace=True)
        X.insert(0, "const", 1)

    if "const" not in candidates:
        candidates = ["const"] + candidates
    else:
        candidates.remove("const")
        candidates = ["const"] + candidates

    X = np.asarray(X[candidates])    
    null_model = np.sum(weights * y) / np.sum(weights)        
    theta_0 = np.append(null_model,  np.zeros(len(candidates)-1))
    g = LinearRegression.score_function(theta_0, X, y, weights)
    Is = - LinearRegression.information_matrix(theta_0, X, weights)
    const = ['const']
    return g, Is, const, candidates


def prepare_step(X, y, candidates, weights, method=None, rename=False):
    duplicates = [item for item, count in collections.Counter(candidates).items() if count > 1]
    candidates = list(dict.fromkeys(candidates))
    X = X.copy()

    if y.name in X.columns.tolist():
        X.drop(y.name, axis=1, inplace=True)
 
    if duplicates:
        print(f"Duplicates variables are found: {duplicates}, and removed. Variables Remaining: {len(candidates) - 1}")

    if rename:
        if "const" in X.columns.tolist():
            candidates_V = [] 
            for i in range(1, len(candidates)):
                candidates_V.append(f"v{i}")
            candidates_V = ["const"] + candidates_V 
        else:
            candidates_V = []
            for i in range(1, len(candidates)+1):
                candidates_V.append(f"v{i+1}")
            candidates_V = ["const"] + candidates_V

        X = X[candidates]

        X.columns = candidates_V
        candidates = candidates_V
    

    if method == "logistic" or method is None:
        g, Is, const, candidates = logistic_null_model(y, X, candidates, weights)
    elif method == "ordinal": 
        g, Is, const, candidates = ordinal_null_model(y, X, candidates)
    elif method == "ols":   
        g, Is, const, candidates = ols_null_model(y, X, candidates, weights)
    
 
    scores = []
    variables = []
    for var in candidates[len(const):]:
 
        model = const + [var]
        loc1 = [candidates.index(v) for v in model]
        score_ = compute_score_submatrix(g, Is, loc1)  
        scores.append(score_)   
        variables.append(var)

    ordered_candidates = const + list(np.array(variables)[np.argsort(scores)[::-1]])
                                
    
   
    return g, Is, X, y, candidates, ordered_candidates

def compute_score_submatrix(g, Is, loc):
    """
    Helper to compute -g' * inv(Is_sub) * g using solve for speed/stability.
    """
    submatrix = Is[np.ix_(loc, loc)]
    subg = g[loc]
    # Instead of inv(submatrix).dot(subg), do np.linalg.solve(submatrix, subg)
    val = np.linalg.solve(submatrix, subg)
    return -np.dot(subg, val)


class Node:
    count: int = 0
    def __init__(self, key, branches, n):
        """
        key: the current subset of variables (excluding 'const')
        branches: how many branches remain
        n: the target subset size or node parameter
        forced_vars: list of variables that must stay in every subset
        """
        self.key = key                # full subset (list of strings)
        self.key2 = key[:n]           # partial subset for bounding
        self.branch_id = n - branches + 1
        self.n = n

        self.child = []
        self.key_list = []
        self.has_branches = branches

    def add_children(self):
        """
        Create child nodes by popping one feature at a time
        but skip if that would drop any forced_var from the subset.
        """
        visit = self.has_branches - 1

        for has_branches_new, _ in enumerate(range(visit, 0, -1)):

            child_branch_id = self.n - has_branches_new - 1
            temp = self.key[:]   

            # Sanity check: child_branch_id might be out of range
            if child_branch_id < 0 or child_branch_id >= len(temp):
                continue

            temp.pop(child_branch_id)

            if len(temp) == self.n - 1:
                continue

            new_node = Node(
                temp, 
                has_branches_new + 2, 
                self.n
            )
            self.child.append(new_node)
            self.key_list.append(temp)

def traverse_tree_best_first(
    X, y, n, nbest, 
    candidates_map=None, 
    candidates=None,
    g=np.array([], np.float64), 
    Is=np.array([], np.float64), 
    forced_vars=None, 
    e={},
    const=['const']
):
    if forced_vars is None:
        forced_vars = []
    else:
        forced_vars = forced_vars[:]
    
    if forced_vars:
        exact_matches = set(var for var in forced_vars if "*" not in var)
        wildcard_matches = [var for var in forced_vars if "*" in var]
    else:
        exact_matches = set()
        wildcard_matches = []

    const = const
    # Root node has all variables except 'const'
    C = candidates[len(const):]
    for fv in forced_vars:
        if not "*" in fv:
            if fv not in C:
                raise ValueError(f"Forced var '{fv}' not found in candidate set!")
    

    root = Node(C, branches=n + 1, n=n)

    # Keep track of the best subsets found so far
    bound = [0]   # This will hold the "worst" among the top-n best scores
    processed_models = []
    processed_scores = []
    count = 0
    bounds = []

    def get_score(model_vars):
        loc1 = [candidates_map.index(c) for c in const] + [candidates_map.index(v) for v in model_vars]
        return compute_score_submatrix(g, Is, loc1)

    def set_bounds(score_val, bound, model_vars):
        """
        Insert score_val into our top-n list if it beats the worst 
        among them. Update 'bound[0]' to the new minimum of the top-n.
        """
        nonlocal count
   
        if len(bounds) < nbest:
            # We haven't filled up our "top-n" set
            bounds.append(score_val)
            bound[0] = min(bounds)  # The worst in top-n
        else:
            # If this new score is better than the current worst
            if score_val > bound[0]:
                # Replace the worst
                idx = bounds.index(bound[0])
                bounds[idx] = score_val
                bound[0] = min(bounds)
        processed_models.append(" ".join(model_vars))
        processed_scores.append(score_val)


    def is_key2_subset_forced_vars(model_vars):
        # Check for exact matches
        exact_check = exact_matches.issubset(set(model_vars))
        
        # Check for wildcard matches
        wildcard_check = all(any(fnmatch.fnmatch(mv, wc) for mv in model_vars) for wc in wildcard_matches)
        
        return exact_check and wildcard_check

    def missing_any_forced_vars(model_vars): 

        missing_exact = exact_matches - set(model_vars)
        missing_wildcards = [
            wc for wc in wildcard_matches 
            if not any(fnmatch.fnmatch(mv, wc) for mv in model_vars) and wc not in model_vars
        ]

        return list(missing_exact) + missing_wildcards
   
    # Priority queue; we store tuples (-score_2, Node)
    # so the node with largest score_2 is expanded first.
    pq = []

    # Evaluate bounding score for the root
    score_2_root = get_score(root.key2)  # partial subset bounding
    # Push into the heap
    heapq.heappush(pq, (-score_2_root, root))



    while pq:
        # Pop the node with the best bounding (largest score_2)
        neg_score_2, cur_node = heapq.heappop(pq)
        score_2 = -neg_score_2
        score_1 = get_score(cur_node.key)
        # print(count+1,  "Key2: ", [str(key) for key in cur_node.key2], "Forced: ",  forced_vars, "Is Subset key2: ", is_key2_subset_forced_vars(cur_node.key2), "Key Missing Forced Features:",  missing_any_forced_vars(cur_node.key))

        if is_key2_subset_forced_vars(cur_node.key2):
            set_bounds(score_2, bound, cur_node.key2)

        count += 1
        # If the bounding for the full set is worse than 
        # our current worst top-n, prune.
        # force the search to continue and expand if bounds is not filled with nbest models

        if len(bounds) >= nbest:
            if score_1 < bound[0]:
                continue  # prune entire branch
        
        # We are also not expanding if any variable is missing from full model
        if any(missing_any_forced_vars(cur_node.key)):
            continue

        # Otherwise, expand
        cur_node.add_children()
        for child in cur_node.child:
            child_score_2 = get_score(child.key2)
            # Push child into priority queue
            heapq.heappush(pq, (-child_score_2, child))
    # print("# Models: ", count, end=' ')
    return processed_models, processed_scores, count

 

def best_subset_bb_logistic_with_priority(
    df, y, nbest, start=1, stop=1, 
    candidates=None, 
    forced_vars=None,
    weights = None,
    normalize=False,
    method = None,

):
   
    if forced_vars is None:
        forced_vars = []

    if method is None: 
        raise ValueError("Method must be specified")
    else:
        if method.lower() == "ordinal" and weights is not None:
            raise ValueError("Weights are not supported for ordinal regression")
        elif method.lower() == "ols":
            warnings.warn("OLS method is experimental and not recommended for production use until further testing is completed")



    if method not in ["logistic", "ordinal", "ols"]:    
        raise ValueError("Method must be one of 'logistic', 'ordinal', or 'ols'")

    
    if weights is None:
        weights = np.ones(len(y))
    else:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        if normalize:
            nobs = df.shape[0]
            weights = (weights / np.sum(weights)) * nobs
   

    g, Is, X, y, candidates_map, ordered_variables = prepare_step(df, y, candidates, weights = weights, method=method)
 
    const =  [item for item in ordered_variables if item not in candidates]

    if stop > len(candidates_map) - 1:
        stop = len(candidates_map) - 1
        print("stop exceeds max # of variables. Reduced to:", stop)

    all_count = 0
    all_models, all_scores, all_varns = [], [], []
    Node.count = 0  

    for n in range(start, stop + 1):
        
        # Do not process if n of forced_vars is greater than n
        if len(forced_vars) > n:
            # print("Skip", n)
            print("Finished Var Family:", n, " Skipped")
            continue
 
        models, scores, count = traverse_tree_best_first(
            X, y, n, nbest=nbest,
            candidates_map=candidates_map,
            candidates=ordered_variables,
            g=g, Is=Is,
            forced_vars=forced_vars,            
            const=const
        )
    
        varn = [n] * len(scores)
        all_count += count
        all_models.extend(models)
        all_scores.extend(scores)
        all_varns.extend(varn)

        print("Finished Var Family:", n)
 

    scores_top, varN_top, models_top = return_top(
        all_scores, 
        all_varns, 
        all_models, 
        nbest=nbest
    )

    result_df = pd.DataFrame({
        "Var Number": varN_top,
        "Models": models_top,
        "Scores": scores_top
    }).sort_values(by=["Var Number","Scores"], ascending=[True,False])

    e = {}
    # print(Node.count)
    return result_df, all_count, e
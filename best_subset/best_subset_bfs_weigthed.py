import heapq
import numpy as np
import pandas as pd
import collections
import itertools

import fnmatch

def loglike(params, X, y, W):
    q = 2 * y - 1
    return np.sum(W * np.log(cdf(q * np.dot(X, params))))

def cdf(X):
    X = np.asarray(X)
    return 1 / (1 + np.exp(-X))

def I(params, X, W):
    X = np.array(X)
 
    L = cdf(np.dot(X, params))
    return -np.dot(W * L * (1 - L) * X.T, X)

def score(params, X, y, W):    
    L = cdf(np.dot(X, params))
    return np.dot(W * (y - L), X)

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



def best_subset_exhaustive_logistic(X, y, candidates, weights=None):
    if weights is None:
        weights = np.ones_like(y)

    
    avg = np.sum(weights * y) / np.sum(weights)
    candidates = candidates[:]

    if "const" not in X.columns.tolist():
        raise NameError("const is missing in X")

    if "const" not in candidates:
        candidates = ["const"] + candidates
    else:
        candidates.remove("const")
        candidates = ["const"] + candidates

    X = X[candidates]

    null_model = np.log(avg / (1 - avg))
    C = candidates[1:]
    theta_0 = np.append(null_model, np.zeros(X.shape[1] - 1))
    g = score(theta_0, X, y, weights)
    Is = I(theta_0, X, weights)

    var_nums = list(range(1, len(candidates[1:]) + 1))
    varN = []
    models = []
    scores = []
    count = 0

    for L in var_nums:
        for model in itertools.combinations(C, L):
            temp_model = list(model)
            var = len(model)
            varN.append(var)

            temp_model.insert(0, "const")
            loc = np.nonzero(np.in1d(candidates, temp_model))[0]
            model_string = " ".join(temp_model[1:])

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



def prepare_step(X, y, candidates, weights ,rename=False):
    duplicates = [item for item, count in collections.Counter(candidates).items() if count > 1]
    candidates = list(dict.fromkeys(candidates))
    X = X.copy()

    if y.name in X.columns.tolist():
        X.drop(y.name, axis=1, inplace=True)
    
    if "const" not  in X.columns.tolist():
        raise NameError("const is missing in X")
    
    else:
        X.drop("const", axis=1, inplace=True)
        X.insert(0, "const", 1)
    
    if "const" not in candidates:
        candidates = ["const"] + candidates
    else:
        candidates.remove("const")
        candidates = ["const"] + candidates

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
    
    
    X = np.asarray(X[candidates])
   
    avg = np.sum(weights*y) / np.sum(weights)    
   
    null_model = np.log(avg / (1 - avg))      
   
    theta_0 = np.append(null_model, np.zeros(X.shape[1] - 1))
  
    g = score(theta_0, X, y, weights)
   
    Is = I(theta_0, X, weights)
 
    scores = []
    variables = []
    for var in candidates[1:]:
 
        model = ['const'] + [var]
        loc1 = [candidates.index(v) for v in model]
        score_ = compute_score_submatrix(g, Is, loc1)  # -g[loc1].T.dot(np.linalg.inv(Is[np.ix_(loc1, loc1)])).dot(g[loc1])
        scores.append(score_)   
        variables.append(var)

    ordered_candidates = ['const'] + list(np.array(variables)[np.argsort(scores)[::-1]])
                                
    
   
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

    def __init__(self, key, branches, n, forced_vars=None):
        """
        key: the current subset of variables (excluding 'const')
        branches: how many branches remain
        n: the target subset size or node parameter
        forced_vars: list of variables that must stay in every subset
        """
        if forced_vars is None:
            forced_vars = []

        self.key = key                # full subset (list of strings)
        self.key2 = key[:n]           # partial subset for bounding
        self.branch_id = n - branches + 1
        self.n = n
        self.forced_vars = forced_vars

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

            # print(temp, self.key2)

            # Sanity check: child_branch_id might be out of range
            if child_branch_id < 0 or child_branch_id >= len(temp):
                continue

            removed_feat = temp.pop(child_branch_id)
            # If removing that feature leads to losing forced var, skip
            if removed_feat in self.forced_vars:
                Node.count += 1
                continue
            

            # Also skip if after removal, any forced var is not in temp
            if not all(fv in temp for fv in self.forced_vars):
                continue

        

            # If we haven't pruned, then child is valid
            # We also skip if it doesn't actually reduce the subset size
            if len(temp) == self.n - 1:
                # This line in the original code was used to skip 
                # same-size sets, but let's keep it for consistency:
                continue

            new_node = Node(
                temp, 
                has_branches_new + 2, 
                self.n, 
                forced_vars=self.forced_vars
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
    weights = None
):
    if forced_vars is None:
        forced_vars = []

    # Root node has all variables except 'const'
    C = candidates[1:]
    for fv in forced_vars:
        if fv not in C:
            raise ValueError(f"Forced var '{fv}' not found in candidate set!")

    root = Node(C, branches=n + 1, n=n, forced_vars=forced_vars)

    # Keep track of the best subsets found so far
    bound = [0]   # This will hold the "worst" among the top-n best scores
    processed_models = []
    processed_scores = []
    count = 0
    bounds = []

    def get_score(model_vars):
        loc1 = [candidates_map.index("const")] + [candidates_map.index(v) for v in model_vars]
        return compute_score_submatrix(g, Is, loc1)

    def set_bounds(score_val, bound, model_vars):
        """
        Insert score_val into our top-n list if it beats the worst 
        among them. Update 'bound[0]' to the new minimum of the top-n.
        """
        nonlocal count

        # Also skip if after removal, any forced var is not in temp


        if count < nbest:
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
        count += 1

    # Priority queue; we store tuples (-score_2, Node)
    # so the node with largest score_2 is expanded first.
    pq = []

    # Evaluate bounding score for the root
    score_2_root = get_score(root.key2)  # partial subset bounding
    # Push into the heap
    heapq.heappush(pq, (-score_2_root, root))

    # We'll prime the system with the partial subset, 
    # ensuring we set initial bounds
    set_bounds(score_2_root, bound, root.key2)

    while pq:
        # Pop the node with the best bounding (largest score_2)
        neg_score_2, cur_node = heapq.heappop(pq)
        score_2 = -neg_score_2

        # Actual full subset score for bounding check
        score_1 = get_score(cur_node.key)

        # If the bounding for the full set is worse than 
        # our current worst top-n, prune.
        if score_1 < bound[0]:
            continue  # prune entire branch

        # Otherwise, expand
        cur_node.add_children()
        for child in cur_node.child:
            # The bounding for the child we use is child's key2, 
            # same logic as original
            child_score_2 = get_score(child.key2)
            # Add child's partial subset to our top-n
            set_bounds(child_score_2, bound, child.key2)
            # Push child into priority queue
            heapq.heappush(pq, (-child_score_2, child))

    return processed_models, processed_scores, count

 

def best_subset_bb_logistic_with_priority(
    df, y, nbest, start=1, stop=1, 
    candidates=None, 
    forced_vars=None,
    weights = None,
    normalize=False
):
   
    if forced_vars is None:
        forced_vars = []
    
    if weights is None:
        weights = np.ones(len(y))
    else:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        if normalize:
            nobs = df.shape[0]
            weights = (weights / np.sum(weights)) * nobs

    g, Is, X, y, candidates_map, ordered_variables = prepare_step(df, y, candidates, weights = weights)

    if stop > len(candidates_map) - 1:
        stop = len(candidates_map) - 1
        print("stop exceeds max # of variables. Reduced to:", stop)

    all_count = 0
    all_models, all_scores, all_varns = [], [], []
    Node.count = 0 
    for n in range(start, stop + 1):
 
        models, scores, count = traverse_tree_best_first(
            X, y, n, nbest=nbest,
            candidates_map=candidates_map,
            candidates=ordered_variables,
            g=g, Is=Is,
            forced_vars=forced_vars,
            weights = weights
        )
    
        varn = [n] * len(scores)
        all_count += count
        all_models.extend(models)
        all_scores.extend(scores)
        all_varns.extend(varn)

        print("Finished Var Family:", n)
    print(nbest)
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
    print(Node.count)
    return result_df, all_count, e
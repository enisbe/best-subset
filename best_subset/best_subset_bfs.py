import heapq
import numpy as np
import pandas as pd
import collections

def loglike(params, X, y):
    q = 2 * y - 1
    X = X
    return np.sum(np.log(cdf(q * np.dot(X, params))))

def cdf(X):
    X = np.asarray(X)
    return 1 / (1 + np.exp(-X))

def I(params, X):
    X = np.array(X)
    L = cdf(np.dot(X, params))
    return -np.dot(L * (1 - L) * X.T, X)

def score(params, X, y):
    y = y
    X = X
    L = cdf(np.dot(X, params))
    return np.dot(y - L, X)

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



def best_subset_exhaustive_logistic(X, y, candidates):
    avg = np.mean(y)
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
    g = score(theta_0, X, y)
    Is = I(theta_0, X)

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



def prepare_step(X, y, candidates, rename=False):
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
    avg = np.mean(y)
    null_model = np.log(avg / (1 - avg))    
    theta_0 = np.append(null_model, np.zeros(X.shape[1] - 1))
    g = score(theta_0, X, y)
   
    Is = I(theta_0, X)

    scores = []
    variables = []
    for var in candidates[1:]:
 
        model = ['const'] + [var]
        loc1 = [candidates.index(v) for v in model]
        score_ = compute_score_submatrix(g, Is, loc1)
        scores.append(score_)   
        variables.append(var)

    ordered_candidates = ['const'] + list(np.array(variables)[np.argsort(scores)[::-1]])
                                
    
   
    return g, Is, X, y, candidates, ordered_candidates

def compute_score_submatrix(g, Is, loc):
    submatrix = Is[np.ix_(loc, loc)]
    subg = g[loc]
    val = np.linalg.solve(submatrix, subg)
    return -np.dot(subg, val)

class Node:
    def __init__(self, key, branches, n, forced_vars=None):
        if forced_vars is None:
            forced_vars = []

        self.key = key
        self.key2 = key[:n]
        self.branch_id = n - branches + 1
        self.n = n
        self.forced_vars = forced_vars

        self.child = []
        self.key_list = []
        self.has_branches = branches

    def add_children(self):
        visit = self.has_branches - 1

        for has_branches_new, _ in enumerate(range(visit, 0, -1)):
            child_branch_id = self.n - has_branches_new - 1
            temp = self.key[:]

            if child_branch_id < 0 or child_branch_id >= len(temp):
                continue

            removed_feat = temp.pop(child_branch_id)
            if removed_feat in self.forced_vars:
                continue

            if not all(fv in temp for fv in self.forced_vars):
                continue

            if len(temp) == self.n - 1:
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
    e={}
):
    if forced_vars is None:
        forced_vars = []

    C = candidates[1:]
    for fv in forced_vars:
        if fv not in C:
            raise ValueError(f"Forced var '{fv}' not found in candidate set!")

    root = Node(C, branches=n + 1, n=n, forced_vars=forced_vars)

    bound = [0]
    processed_models = []
    processed_scores = []
    count = 0
    bounds = []

    def get_score(model_vars):
        loc1 = [candidates_map.index("const")] + [candidates_map.index(v) for v in model_vars]
        return compute_score_submatrix(g, Is, loc1)

    def set_bounds(score_val, bound, model_vars):
        nonlocal count
        if count < nbest:
            bounds.append(score_val)
            bound[0] = min(bounds)
        else:
            if score_val > bound[0]:
                idx = bounds.index(bound[0])
                bounds[idx] = score_val
                bound[0] = min(bounds)
        processed_models.append(" ".join(model_vars))
        processed_scores.append(score_val)
        count += 1

    pq = []

    score_2_root = get_score(root.key2)
    heapq.heappush(pq, (-score_2_root, root))

    set_bounds(score_2_root, bound, root.key2)

    while pq:
        neg_score_2, cur_node = heapq.heappop(pq)
        score_2 = -neg_score_2

        score_1 = get_score(cur_node.key)

        if score_1 < bound[0]:
            continue

        cur_node.add_children()
        for child in cur_node.child:
            child_score_2 = get_score(child.key2)
            set_bounds(child_score_2, bound, child.key2)
            heapq.heappush(pq, (-child_score_2, child))

    return processed_models, processed_scores, count


def best_subset_bb_logistic_with_priority(
    df, y, nbest, start=1, stop=1, 
    candidates=None, 
    forced_vars=None
):
    if forced_vars is None:
        forced_vars = []

    g, Is, X, y, candidates_map, ordered_variables = prepare_step(df, y, candidates)

    if stop > len(candidates_map) - 1:
        stop = len(candidates_map) - 1
        print("stop exceeds max # of variables. Reduced to:", stop)

    all_count = 0
    all_models, all_scores, all_varns = [], [], []

    for n in range(start, stop + 1):
  
        models, scores, count = traverse_tree_best_first(
            X, y, n, nbest=nbest,
            candidates_map=candidates_map,
            candidates=ordered_variables,
            g=g, Is=Is,
            forced_vars=forced_vars
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
    return result_df, all_count, e

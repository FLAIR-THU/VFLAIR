def mismatch_preds(node, original_y):
    noised_val = node.val.copy()
    original_val = [0] * node.num_classes

    for r in range(node.row_count):
        original_val[int(original_y[node.idxs[r]]) - 1] += 1 / float(node.row_count)

    noised_argmax, original_argmax = 0, 0
    max_val = 0

    for i in range(len(noised_val)):
        if noised_val[i] > max_val:
            max_val = noised_val[i]
            noised_argmax = i

    max_val = 0
    for i in range(len(original_val)):
        if original_val[i] > max_val:
            max_val = original_val[i]
            original_argmax = i

    return noised_argmax != original_argmax


def mark_contaminated_nodes(node, target_nodes, original_y):
    if node.is_leaf():
        node.is_all_subsequent_children_contaminated = mismatch_preds(node, original_y)
        return
    else:
        mark_contaminated_nodes(node.left, target_nodes, original_y)
        mark_contaminated_nodes(node.right, target_nodes, original_y)
        if node.left.is_all_subsequent_children_contaminated or node.right.is_all_subsequent_children_contaminated:
            if mismatch_preds(node, original_y):
                node.is_all_subsequent_children_contaminated = True
            else:
                target_nodes.append(node)


def grafting_tree(tree, original_y):
    root_of_problems = []
    mark_contaminated_nodes(tree.dtree, root_of_problems, original_y)

    for node in root_of_problems:
        node.best_party_id = -1
        node.best_col_id = -1
        node.best_threshold_id = -1
        node.best_score = -1 * float('-inf')
        node.is_leaf_flag = -1
        node.is_pure_flag = -1

        node.use_only_active_party = True
        node.secure_flag_exclude_passive_parties = True
        node.y = original_y
        node.entire_class_cnt = [0] * node.num_classes
        node.entire_datasetsize = len(node.y)
        for i in range(node.entire_datasetsize):
            node.entire_class_cnt[int(node.y[i])] += 1.0

        node.giniimp = node.compute_giniimp()
        node.val = node.compute_weight()
        best_split = node.find_split()

        if node.is_leaf():
            node.is_leaf_flag = 1
        else:
            node.is_leaf_flag = 0

        if node.is_leaf_flag == 0:
            node.party_id = best_split[0]
            if node.party_id != -1:
                node.record_id = node.parties[node.party_id].insert_lookup_table(best_split[1], best_split[2])
                node.make_children_nodes(best_split[0], best_split[1], best_split[2])
            else:
                node.is_leaf_flag = 1


def grafting_forest(clf, original_y):
    for tree in clf.estimators:
        grafting_tree(tree, original_y)

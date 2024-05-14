import networkx as nx
import numpy as np
import os
import pickle
import torch
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


def process_features(features):
    row_sum_diag = np.sum(features, axis=1)
    row_sum_diag_inv = np.power(row_sum_diag, -1)
    row_sum_diag_inv[np.isinf(row_sum_diag_inv)] = 0.
    row_sum_inv = np.diag(row_sum_diag_inv)
    return np.dot(row_sum_inv, features)


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data1(dataset):
    ## get data
    if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
        data_path = '../../../share_dataset/Cora/'
        suffixs = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph']
        objects = []
        for suffix in suffixs:
            file = os.path.join(data_path, 'ind.%s.%s' % (dataset, suffix))
            objects.append(pickle.load(open(file, 'rb'), encoding='latin1'))
        x, y, allx, ally, tx, ty, graph = objects
        x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()

        # test indices
        test_index_file = os.path.join(data_path, 'ind.%s.test.index' % dataset)
        with open(test_index_file, 'r') as f:
            lines = f.readlines()
        indices = [int(line.strip()) for line in lines]
        min_index, max_index = min(indices), max(indices)

        # preprocess test indices and combine all data
        tx_extend = np.zeros((max_index - min_index + 1, tx.shape[1]))
        features = np.vstack([allx, tx_extend])
        features[indices] = tx
        ty_extend = np.zeros((max_index - min_index + 1, ty.shape[1]))
        labels = np.vstack([ally, ty_extend])
        labels[indices] = ty
        labels1 = []
        for i in range(len(labels)):
            labels1.append(labels[i].argmax())
        labels1 = np.array(labels1)
        # get adjacency matrix
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()
        # adj = torch.from_numpy(adj)
        adj = np.array(adj)
        idx_train = np.arange(0, len(y), 1)
        idx_val = np.arange(len(y), len(y) + 500, 1)
        idx_test = np.array(indices)


    elif dataset == 'polblogs':
        adj = np.zeros((1222, 1222))
        with open('data/' + str(dataset) + '.txt') as f:
            for j in f:
                entry = [float(x) for x in j.split(" ")]
                adj[int(entry[0]), int(entry[1])] = 1
                adj[int(entry[1]), int(entry[0])] = 1
        labels1 = np.loadtxt('data/' + str(dataset) + '_label.txt')
        labels1 = labels1.astype(int)
        labels1 = labels1[:, 1:].flatten()
        idx_train = np.loadtxt('data/' + str(dataset) + '_train_node.txt')
        idx_train = idx_train.astype(int)
        idx_val = np.loadtxt('data/' + str(dataset) + '_validation_node.txt')
        idx_val = idx_val.astype(int)
        idx_test = np.loadtxt('data/' + str(dataset) + '_test_node.txt')
        idx_test = idx_test.astype(int)

        features = np.eye(adj.shape[0])

    elif dataset == 'cora_ml':
        filename = 'data/' + str(dataset) + '_adj' + '.npz'
        adj = sp.load_npz(filename)
        filename = 'data/' + str(dataset) + '_features' + '.npz'
        features = sp.load_npz(filename)
        filename = 'data/' + str(dataset) + '_label' + '.npy'
        labels1 = np.load(filename)
        filename = 'data/' + str(dataset) + '_train_node' + '.npy'
        idx_train = np.load(filename)
        filename = 'data/' + str(dataset) + '_val_node' + '.npy'
        idx_val = np.load(filename)
        filename = 'data/' + str(dataset) + '_test_node' + '.npy'
        idx_test = np.load(filename)

    else:

        filename = 'data/' + 'amazon_electronics_photo' + '_adj' + '.npz'
        adj = sp.load_npz(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_features' + '.npz'
        features = sp.load_npz(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_label' + '.npy'
        labels1 = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_train_node' + '.npy'
        idx_train = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_val_node' + '.npy'
        idx_val = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_test_node' + '.npy'
        idx_test = np.load(filename)
        filename = 'data/' + 'amazon_electronics_photo' + '_label' + '.npy'
        labels1 = np.load(filename)

    return sp.csr_matrix(adj), sp.csr_matrix(features), idx_train, idx_val, idx_test, labels1


def get_adj(filename, require_lcc=True):
    adj, features, labels = load_npz(filename)
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1

    if require_lcc:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels = labels[lcc]
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    # whether to set diag=0?
    adj.setdiag(0)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()

    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

    return adj, features, labels


def load_npz(file_name, is_sparse=True):
    with np.load(file_name) as loader:
        # loader = dict(loader)
        if is_sparse:
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                          loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None
            labels = loader.get('labels')
        else:
            adj = loader['adj_data']
            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None
            labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)
    return adj, features, labels


def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def largest_connected_components(adj, n_components=1):
    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # splits are random

    # Remove diagonal elements
    # print('adj', adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # print('adj', adj, adj.shape)
    # Check the diagonal elements are zeros:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # Sparse matrix with DIAgonal storage
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    # print('edges', edges, edges.shape)
    # print('edges_all', edges_all, edges_all.shape)

    # Link index
    num_val = int(edges.shape[0] * 0.1)
    num_test = int(edges.shape[0] * 0.6)
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # print('test_edges', test_edges, test_edges.shape)
    # print('val_edges', val_edges, val_edges.shape)
    # print('train_edges', train_edges, train_edges.shape)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue  # Break this cycle
        if ismember([idx_i, idx_j], np.array(edges_all)):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], np.array(edges_all)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)  # ~is Negate
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(train_edges, val_edges)
    assert ~ismember(val_edges, test_edges)
    assert ~ismember(train_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_trian = adj_train + adj_train.T

    # Note: these edge lists only contain sigle direction of edge!
    return adj_trian, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def split_features(args, feature):
    # num_feature_per_party = int(feature.shape[1]/args.num_parties)
    feature = np.array(feature.todense())
    if feature.shape[1] % args.num_parties != 0:
        del_v = np.random.randint(0, feature.shape[1], 1 * feature.shape[1] % args.num_parties)
        del_v = list(del_v)
        feature = np.delete(feature, del_v, axis=1)
        # expend_arr = np.zeros((feature.shape[0], int(feature.shape[1]//args.num_parties-feature.shape[1]%args.num_parties)))
        # feature = np.hstack([feature, expend_arr])

    # np.random.sample(range(0, feature.shape[1]), num_feature_per_party)
    featureT = feature.T
    np.random.shuffle(featureT)
    feat = featureT.T
    feature_list = np.split(feat, args.num_parties, axis=1)
    feature_list = [sp.csr_matrix(feat) for feat in feature_list]
    return feature_list


def split_graph(args, adj, feature, split_method, split_ratio=0.5, with_s=True, with_f=False):
    # Create new graph graph_A, graph_B
    # Function to build test set with 10% positive links
    # splits are random
    degrees = np.array(adj.todense()).sum(0)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # print('adj', adj, adj.shape)
    # Check the diagonal elements are zeros:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # Sparse matrix with DIAgonal storage
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]

    # print('edges', edges, edges.shape)
    # print('edges_all', edges_all, edges_all.shape)

    # Link index
    if with_s == True:
        if split_method == 'com':
            num_graph_A = int(edges.shape[0] * split_ratio)
            all_edge_idx = list(range(edges.shape[0]))
            np.random.shuffle(all_edge_idx)
            A_edge_idx = all_edge_idx[:num_graph_A]

            A_edges = edges[A_edge_idx]
            B_edges = np.delete(edges, A_edge_idx, axis=0)

        elif split_method == 'alone':
            num_graph_A = int(edges.shape[0] * split_ratio)
            num_graph_B = int(edges.shape[0] * (1 - split_ratio))
            all_edge_idx = list(range(edges.shape[0]))

            np.random.shuffle(all_edge_idx)
            A_edge_idx = all_edge_idx[:num_graph_A]
            np.random.shuffle(all_edge_idx)
            B_edge_idx = all_edge_idx[:num_graph_B]

            A_edges = edges[A_edge_idx]
            B_edges = edges[B_edge_idx]


        elif split_method == 'abs':
            A_edge_idx = []
            num_graph_A = int(edges.shape[0] * split_ratio)
            all_edge_idx = list(range(edges.shape[0]))
            np.random.shuffle(all_edge_idx)
            for i in all_edge_idx:
                if (degrees[edges[i][0]] >= 2 and degrees[edges[i][1]] >= 2):
                    A_edge_idx.append(i)
                    degrees[edges[i][0]] -= 1
                    degrees[edges[i][1]] -= 1
                if len(A_edge_idx) == num_graph_A:
                    break

            A_edges = edges[A_edge_idx]
            B_edges = np.delete(edges, A_edge_idx, axis=0)

        print(len(degrees.nonzero()[0]))

        data_A = np.ones(A_edges.shape[0])
        data_B = np.ones(B_edges.shape[0])

        # Re-build adj matrix
        adj_A = sp.csr_matrix((data_A, (A_edges[:, 0], A_edges[:, 1])), shape=adj.shape)
        adj_A = adj_A + adj_A.T
        degree_A = np.array(adj_A.sum(0))
        print(len(degree_A.nonzero()[0]))
        adj_B = sp.csr_matrix((data_B, (B_edges[:, 0], B_edges[:, 1])), shape=adj.shape)
        adj_B = adj_B + adj_B.T
        degree_B = np.array(adj_B.sum(0))
        print(len(degree_B.nonzero()[0]))
    else:
        adj_A = sp.csr_matrix(np.eye(adj.shape[0]))
        adj_B = sp.csr_matrix(np.eye(adj.shape[0]))
    # Feature split evenly
    # feature_A = torch.split(feature, feature.size()[1] // 2, dim=1)[0]
    # feature_B = torch.split(feature, feature.size()[1] // 2, dim=1)[1]
    if with_f == True:
        X_NUM = int(feature.shape[1] // 2)
        feature = np.array(feature.todense())
        feature_A = feature[:, :X_NUM]
        feature_B = feature[:, X_NUM:2 * X_NUM]
    else:
        feature_A = np.eye(adj.shape[0])
        feature_B = np.eye(adj.shape[0])
    # feature_A = feature
    # feature_B = feature

    # adj_B_tui = adj_B
    # adj_A = preprocess_adj(adj_A)
    # adj_A = sparse_mx_to_torch_sparse_tensor(adj_A)

    # adj_B = preprocess_adj(adj_B)
    # adj_B = sparse_mx_to_torch_sparse_tensor(adj_B)
    if args.dataset == 'polblogs':
        feature_A = np.eye(adj.shape[0])
        feature_B = np.eye(adj.shape[0])

    return adj_A, adj_B, sp.csr_matrix(feature_A), sp.csr_matrix(feature_B)


def split_graph1(args, adj, feature, split_method, split_ration=0.5):
    # Create new graph graph_A, graph_B
    # Function to build test set with 10% positive links
    # splits are random

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # print('adj', adj, adj.shape)
    # Check the diagonal elements are zeros:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)  # Sparse matrix with DIAgonal storage
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]

    # print('edges', edges, edges.shape)
    # print('edges_all', edges_all, edges_all.shape)

    # Link index
    if split_method == 'com':
        num_graph_A = int(edges.shape[0] * split_ration)
        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)
        A_edge_idx = all_edge_idx[:num_graph_A]

        A_edges = edges[A_edge_idx]
        B_edges = np.delete(edges, A_edge_idx, axis=0)

    elif split_method == 'alone':
        num_graph_A = int(edges.shape[0] * split_ration)
        num_graph_B = int(edges.shape[0] * (1 - split_ration))
        all_edge_idx = list(range(edges.shape[0]))

        np.random.shuffle(all_edge_idx)
        A_edge_idx = all_edge_idx[:num_graph_A]
        np.random.shuffle(all_edge_idx)
        B_edge_idx = all_edge_idx[:num_graph_B]

        A_edges = edges[A_edge_idx]
        B_edges = edges[B_edge_idx]

    data_A = np.ones(A_edges.shape[0])
    data_B = np.ones(B_edges.shape[0])

    # Re-build adj matrix
    adj_A = sp.csr_matrix((data_A, (A_edges[:, 0], A_edges[:, 1])), shape=adj.shape)
    adj_A = adj_A + adj_A.T

    adj_B = sp.csr_matrix((data_B, (B_edges[:, 0], B_edges[:, 1])), shape=adj.shape)
    adj_B = adj_B + adj_B.T

    # Feature split evenly

    # feature_A = torch.split(feature, feature.size()[1] // 2, dim=1)[0]
    # feature_B = torch.split(feature, feature.size()[1] // 2, dim=1)[1]

    X_NUM = int(feature.shape[1] // 2)
    feature = np.array(feature.todense())
    feature_A = feature[:, :X_NUM]
    feature_B = feature[:, X_NUM:2 * X_NUM]
    # feature_A = feature
    # feature_B = feature

    # adj_B_tui = adj_B
    # adj_A = preprocess_adj(adj_A)
    # adj_A = sparse_mx_to_torch_sparse_tensor(adj_A)

    # adj_B = preprocess_adj(adj_B)
    # adj_B = sparse_mx_to_torch_sparse_tensor(adj_B)

    return adj_A, adj_B, sp.csr_matrix(feature_A), sp.csr_matrix(feature_B)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    # print('adj22',adj,type(adj))
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


def accuracy(pred, targ):
    pred = torch.max(pred, 1)[1]
    ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    return ac


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

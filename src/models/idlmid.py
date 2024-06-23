import math

eps = 1e-15


def is_satisfied_mi_bound(num_classes, xi, num_idxs_within_node, num_row, entire_class_cnt, prior,
                          y_class_cnt_within_node):
    in_kl_divergence = 0
    out_kl_divergence = 0

    if xi > 0:
        for c in range(num_classes):
            nc_div_n = y_class_cnt_within_node[c] / float(num_idxs_within_node)
            Nc_div_N = prior[c]
            Nc_m_nc_div_N_m_n = (entire_class_cnt[c] - y_class_cnt_within_node[c]) / float(
                num_row - num_idxs_within_node)

            in_kl_divergence += nc_div_n * math.log(eps + nc_div_n / Nc_div_N)
            out_kl_divergence += Nc_m_nc_div_N_m_n * math.log(eps + Nc_m_nc_div_N_m_n / Nc_div_N)

        return max(in_kl_divergence, out_kl_divergence) <= xi
    else:
        return True

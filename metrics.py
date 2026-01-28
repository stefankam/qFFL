import argparse

import numpy as np

EPS = 1e-12


def gini(x):
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return 0.0
    values = values.flatten()
    if np.all(values == 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    cumulative = np.cumsum(sorted_vals)
    gini_coeff = (n + 1 - 2 * np.sum(cumulative) / (cumulative[-1] + EPS)) / n
    return float(gini_coeff)


def jain(x):
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return 0.0
    numerator = np.square(np.sum(values))
    denominator = values.size * np.sum(np.square(values)) + EPS
    return float(numerator / denominator)


def _print_metrics(metrics):
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.tolist()}")
        else:
            print(f"{key}: {value}")


def _normalize_selected(selected, num_clients):
    if not selected:
        return selected
    if isinstance(selected[0], (list, tuple, np.ndarray)):
        first = selected[0]
        if len(first) == 0:
            return selected
        if isinstance(first[0], str):
            mapping = {f"client_{i:05d}": i for i in range(num_clients)}
            normalized = []
            for round_sel in selected:
                normalized.append([mapping[cid] for cid in round_sel])
            return normalized
    return selected


def compute_metrics(
    rounds,
    num_clients,
    m_per_round,
    availability,
    selected,
    loss_before,
    loss_after,
    eps=EPS,
    accuracy=None,
    accuracy_final=None,
    accuracy_train=None,
    accuracy_train_final=None,
    group_id=None,
    p_star=None,
    print_metrics=False,
):
    availability_arr = np.asarray(availability, dtype=float)
    pi_k = availability_arr.mean(axis=0)

    selection_counts = np.zeros(num_clients, dtype=float)
    selected = _normalize_selected(selected, num_clients)
    for t in range(rounds):
        for client_id in selected[t]:
            selection_counts[client_id] += 1

    loss_before_arr = np.asarray(loss_before, dtype=float)
    loss_after_arr = np.asarray(loss_after, dtype=float)
    delta_u = loss_before_arr - loss_after_arr
    valid = np.isfinite(delta_u)
    delta_u = np.where(valid, delta_u, 0.0)
    u_k = np.sum(delta_u, axis=0)
    u_tilde_k = u_k / (pi_k + eps)

    metrics = {
        "pi_k": pi_k,
        "S_k": selection_counts,
        "u_k": u_k,
        "u_tilde_k": u_tilde_k,
    }

    acc_values = None
    if accuracy_final is not None:
        acc_values = np.asarray(accuracy_final, dtype=float)
    elif accuracy is not None:
        accuracy_arr = np.asarray(accuracy, dtype=float)
        if accuracy_arr.ndim == 2:
            valid_rows = np.where(np.any(np.isfinite(accuracy_arr), axis=1))[0]
            if valid_rows.size > 0:
                acc_values = accuracy_arr[valid_rows[-1]]
        else:
            acc_values = accuracy_arr

    train_acc_values = None
    try:
        train_final = accuracy_train_final
    except NameError:
        train_final = None
    try:
        train_series = accuracy_train
    except NameError:
        train_series = None

    if train_final is not None:
        train_acc_values = np.asarray(train_final, dtype=float)
    elif train_series is not None:
        train_acc_arr = np.asarray(train_series, dtype=float)
        if train_acc_arr.ndim == 2:
            valid_rows = np.where(np.any(np.isfinite(train_acc_arr), axis=1))[0]
            if valid_rows.size > 0:
                train_acc_values = train_acc_arr[valid_rows[-1]]
        else:
            train_acc_values = train_acc_arr

    if acc_values is not None:
        mean_acc = float(np.nanmean(acc_values))
        acc_var = float(np.nanvar(acc_values))
        jain_acc = jain(np.nan_to_num(acc_values, nan=0.0))
        metrics["mean_acc"] = mean_acc
        metrics["Avg_Acc"] = mean_acc
        metrics["Acc_Var"] = acc_var
        metrics["Var_acc_across_clients_final"] = acc_var
        metrics["Jain_Acc"] = jain_acc
    else:
        metrics["mean_acc"] = float("nan")
        metrics["Avg_Acc"] = float("nan")
        metrics["Acc_Var"] = float("nan")
        metrics["Var_acc_across_clients_final"] = float("nan")
        metrics["Jain_Acc"] = float("nan")

    if train_acc_values is not None:
        train_mean = float(np.nanmean(train_acc_values))
        train_var = float(np.nanvar(train_acc_values))
        train_jain = jain(np.nan_to_num(train_acc_values, nan=0.0))
        metrics["mean_train_acc"] = train_mean
        metrics["Avg_Train_Acc"] = train_mean
        metrics["Train_Acc_Var"] = train_var
        metrics["Var_train_acc_across_clients_final"] = train_var
        metrics["Jain_Train_Acc"] = train_jain
    else:
        metrics["mean_train_acc"] = float("nan")
        metrics["Avg_Train_Acc"] = float("nan")
        metrics["Train_Acc_Var"] = float("nan")
        metrics["Var_train_acc_across_clients_final"] = float("nan")
        metrics["Jain_Train_Acc"] = float("nan")

    mean_u_tilde = float(np.mean(u_tilde_k))
    std_u_tilde = float(np.std(u_tilde_k))
    metrics["CV_u_tilde"] = std_u_tilde / (mean_u_tilde + eps)
    metrics["Jain_u_tilde"] = jain(u_tilde_k)

    sel_gap = np.abs(selection_counts / m_per_round - rounds / num_clients)
    metrics["SelGap_L1"] = float(np.sum(sel_gap) / rounds)
    metrics["SelGap_Linf"] = float(np.max(sel_gap))
    metrics["Gini_S"] = gini(selection_counts)

    if group_id is not None and p_star is not None:
        group_id_arr = np.asarray(group_id, dtype=int)
        p_star_arr = np.asarray(p_star, dtype=float)
        group_counts = np.zeros_like(p_star_arr, dtype=float)
        for k in range(num_clients):
            group_counts[group_id_arr[k]] += selection_counts[k]
        total_selected = np.sum(selection_counts)
        p_hat = group_counts / (total_selected + eps)
        metrics["max_group_gap"] = float(np.max(np.abs(p_hat - p_star_arr)))
        metrics["KL_p_hat_p_star"] = float(
            np.sum(p_hat * np.log((p_hat + eps) / (p_star_arr + eps)))
        )

    if print_metrics:
        _print_metrics(metrics)

    return metrics


def load_inputs_npz(path):
    data = np.load(path, allow_pickle=True)
    selected = data["selected"].tolist()
    return {
        "rounds": int(data["rounds"]),
        "num_clients": int(data["num_clients"]),
        "m_per_round": int(data["m_per_round"]),
        "availability": data["availability"],
        "selected": selected,
        "loss_before": data["loss_before"],
        "loss_after": data["loss_after"],
        "accuracy": data.get("accuracy"),
        "accuracy_final": data.get("accuracy_final"),
        "accuracy_train": data.get("accuracy_train"),
        "accuracy_train_final": data.get("accuracy_train_final"),
        "group_id": data.get("group_id"),
        "p_star": data.get("p_star"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Path to npz with metrics inputs.")
    args = parser.parse_args()

    inputs = load_inputs_npz(args.npz)
    compute_metrics(
        rounds=inputs["rounds"],
        num_clients=inputs["num_clients"],
        m_per_round=inputs["m_per_round"],
        availability=inputs["availability"],
        selected=inputs["selected"],
        loss_before=inputs["loss_before"],
        loss_after=inputs["loss_after"],
        accuracy=inputs["accuracy"],
        accuracy_final=inputs["accuracy_final"],
        accuracy_train=inputs["accuracy_train"],
        accuracy_train_final=inputs["accuracy_train_final"],
        group_id=inputs["group_id"],
        p_star=inputs["p_star"],
        print_metrics=True,
    )


if __name__ == "__main__":
    main()


__all__ = ["compute_metrics", "gini", "jain", "load_inputs_npz"]

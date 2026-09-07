from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

OBS = DATA / 'Qatraneh_R15R4_Observations.csv'
LOCKED = DATA / 'Qatraneh_R15R4_NestedModelDecisions_LOCKED.csv'

# Explicit reproducibility choice for the independently reconstructed E search.
# The manuscript-stated bounds are retained; the grid is logarithmic and dense.
N_LAMBDA = 501
WINDOWS = [6, 7, 8, 9, 10, 12, 14]
TAIL_WINDOWS = [6, 7, 8, 9, 10, 12]
MODEL_ORDER = {'G': 0, 'P': 1, 'E': 2}

ROUTE_COLS = {
    'Field K': 'Kfield_x1e-6_SI',
    'Fine LF': 'Fine_LF_x1e-8_m3kg',
    'Fine HF': 'Fine_HF_x1e-8_m3kg',
    'Coarse LF': 'Coarse_LF_x1e-8_m3kg',
    'Coarse HF': 'Coarse_HF_x1e-8_m3kg',
}


def _linfit(X, y):
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    return coef, float(resid @ resid)


def fit_G(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    return _linfit(X, y)


def pred_G(params, x):
    return params[0] + params[1] * np.asarray(x, float)


def fit_P(x, y):
    """Fit P(d)=alpha+beta*min(d,tau), beta>=0, with tau chosen by training SSE."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    best = None
    for tau in sorted(set(x)):
        if np.sum(x <= tau) < 3 or np.sum(x > tau) < 2:
            continue
        z = np.minimum(x, tau)
        X = np.column_stack([np.ones(len(x)), z])
        coef, sse = _linfit(X, y)
        # With only beta constrained, if the unconstrained solution violates beta>=0,
        # the constrained optimum lies on beta=0 (the constant boundary model).
        if coef[1] < 0:
            coef = np.array([float(y.mean()), 0.0])
            sse = float(np.sum((y - y.mean()) ** 2))
        cand = (sse, float(tau), coef)
        if best is None or cand[0] < best[0] - 1e-12 or (
            abs(cand[0] - best[0]) <= 1e-12 and cand[1] < best[1]
        ):
            best = cand
    if best is None:
        raise ValueError('P is not admissible for this training set')
    sse, tau, coef = best
    return (coef, tau), sse


def pred_P(params, x):
    coef, tau = params
    return coef[0] + coef[1] * np.minimum(np.asarray(x, float), tau)


def fit_E(x, y, n_lambda=N_LAMBDA):
    """Fit E(d)=alpha+beta*d+A*exp(-d/lambda), A>=0, lambda>0.

    The lambda candidates are evaluated in a vectorized batched least-squares
    calculation. This is numerically equivalent to the earlier per-lambda loop
    but makes the full reproducibility and jackknife checks substantially faster.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xs = np.sort(np.unique(x))
    if len(xs) < 2:
        raise ValueError('E requires at least two distinct distances')
    min_spacing = float(np.min(np.diff(xs)))
    span = float(np.max(x) - np.min(x))
    lo = 0.5 * min_spacing
    hi = 10.0 * span
    lambdas = np.geomspace(lo, hi, n_lambda)

    e = np.exp(-x[None, :] / lambdas[:, None])
    n = len(x)
    sx = float(np.sum(x))
    sx2 = float(x @ x)
    sy = float(np.sum(y))
    sxy = float(x @ y)
    se = np.sum(e, axis=1)
    sxe = e @ x
    see = np.sum(e * e, axis=1)
    sey = e @ y

    normal = np.empty((n_lambda, 3, 3), float)
    normal[:, 0, 0] = n
    normal[:, 0, 1] = sx
    normal[:, 1, 0] = sx
    normal[:, 1, 1] = sx2
    normal[:, 0, 2] = se
    normal[:, 2, 0] = se
    normal[:, 1, 2] = sxe
    normal[:, 2, 1] = sxe
    normal[:, 2, 2] = see
    rhs = np.column_stack([np.full(n_lambda, sy), np.full(n_lambda, sxy), sey])

    try:
        coef = np.linalg.solve(normal, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        coef = np.vstack([
            np.linalg.lstsq(np.column_stack([np.ones(n), x, e[j]]), y, rcond=None)[0]
            for j in range(n_lambda)
        ])

    pred = coef[:, 0, None] + coef[:, 1, None] * x[None, :] + coef[:, 2, None] * e
    sse = np.sum((y[None, :] - pred) ** 2, axis=1)

    # With only A constrained, if unconstrained A<0 the constrained optimum
    # is A=0, reducing E to the fitted gradient boundary.
    bad = coef[:, 2] < 0
    if np.any(bad):
        gcoef, gsse = fit_G(x, y)
        coef[bad, 0] = gcoef[0]
        coef[bad, 1] = gcoef[1]
        coef[bad, 2] = 0.0
        sse[bad] = gsse

    j = int(np.argmin(sse))
    return (coef[j], float(lambdas[j])), float(sse[j])


def pred_E(params, x):
    coef, lam = params
    x = np.asarray(x, float)
    return coef[0] + coef[1] * x + coef[2] * np.exp(-x / lam)


def fit_model(model, x, y):
    if model == 'G':
        return fit_G(x, y)[0]
    if model == 'P':
        return fit_P(x, y)[0]
    if model == 'E':
        return fit_E(x, y)[0]
    raise KeyError(model)


def predict(model, params, x):
    if model == 'G':
        return pred_G(params, x)
    if model == 'P':
        return pred_P(params, x)
    if model == 'E':
        return pred_E(params, x)
    raise KeyError(model)


def loocv_scores(x, y, models=('G', 'P', 'E')):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    preds = {m: [] for m in models}
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        xt, yt = x[keep], y[keep]
        for model in models:
            params = fit_model(model, xt, yt)
            preds[model].append(float(predict(model, params, [x[i]])[0]))
    scores = {
        m: float(np.sqrt(np.mean((y - np.asarray(preds[m])) ** 2)))
        for m in models
    }
    winner = min(scores, key=lambda m: (scores[m], MODEL_ORDER[m]))
    return scores, winner


def dstar_from_sequence(seq_rows):
    full = seq_rows.iloc[-1]['Winner']
    wins = seq_rows['Winner'].tolist()
    ws = seq_rows['Outer_window_m'].tolist()
    for i, w in enumerate(ws):
        if all(m == full for m in wins[i:]):
            return float(w)
    return float(ws[-1])


def route_data(obs):
    for side in ['North', 'South']:
        sdf = obs[obs['Side'] == side].sort_values('Distance_m')
        x = sdf['Distance_m'].to_numpy(float)
        for route, col in ROUTE_COLS.items():
            yield side, route, x, sdf[col].to_numpy(float)


def main():
    obs = pd.read_csv(OBS)
    locked = pd.read_csv(LOCKED) if LOCKED.exists() else None

    # 1) Full nested LOOCV score record and winners.
    score_rows = []
    for side, route, x_all, y_all in route_data(obs):
        for w in WINDOWS:
            mask = x_all <= w
            scores, winner = loocv_scores(x_all[mask], y_all[mask])
            ordered = sorted(scores.items(), key=lambda kv: (kv[1], MODEL_ORDER[kv[0]]))
            margin_pct = 100.0 * (ordered[1][1] - ordered[0][1]) / ordered[0][1]
            score_rows.append({
                'Side': side,
                'Route': route,
                'Outer_window_m': w,
                'Winner': winner,
                'LOOCV_RMSE_G': scores['G'],
                'LOOCV_RMSE_P': scores['P'],
                'LOOCV_RMSE_E': scores['E'],
                'Runner_up': ordered[1][0],
                'Winner_margin_pct': margin_pct,
            })
    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(DATA / 'Qatraneh_R16_GEP_LOOCV_Scores_RECOMPUTED.csv', index=False)

    # D* and complete-model summary from the recomputed sequence.
    decision_rows = []
    route_summary = []
    for (side, route), q in scores_df.groupby(['Side', 'Route'], sort=False):
        q = q.sort_values('Outer_window_m')
        full = q.iloc[-1]['Winner']
        dstar = dstar_from_sequence(q)
        for _, r in q.iterrows():
            decision_rows.append({
                'Side': side,
                'Route': route,
                'Outer_window_m': int(r['Outer_window_m']),
                'Model': r['Winner'],
                'Full_model': full,
                'Dstar_m': dstar,
            })
        fullrow = q.iloc[-1]
        route_summary.append({
            'Side': side,
            'Route': route,
            'Full_model': full,
            'Dstar_m': dstar,
            'Full_winner_margin_pct': float(fullrow['Winner_margin_pct']),
        })
    decisions_df = pd.DataFrame(decision_rows)
    decisions_df.to_csv(DATA / 'Qatraneh_R16_NestedModelDecisions_RECOMPUTED.csv', index=False)
    summary_df = pd.DataFrame(route_summary)
    summary_df.to_csv(DATA / 'Qatraneh_R16_ModelMarginSummary.csv', index=False)

    # 2) Verify against the previous locked analytical record.
    match_count = None
    if locked is not None:
        m = locked.merge(
            decisions_df,
            on=['Side', 'Route', 'Outer_window_m'],
            suffixes=('_locked', '_recomputed')
        )
        m['Match'] = m['Model_locked'] == m['Model_recomputed']
        m.to_csv(DATA / 'Qatraneh_R16_Locked_vs_Recomputed.csv', index=False)
        match_count = int(m['Match'].sum())
        if match_count != len(m):
            bad = m.loc[~m['Match'], ['Side', 'Route', 'Outer_window_m', 'Model_locked', 'Model_recomputed']]
            raise RuntimeError(f'Independent reimplementation mismatch:\n{bad}')

    # 3) C10 threshold sensitivity, including the threshold-free exact-maximum result.
    c_rows = []
    c_route_rows = []
    for side, route, x_all, y_all in route_data(obs):
        c10 = float(np.max(y_all[x_all <= 10]) / np.max(y_all))
        dstar = float(summary_df[(summary_df.Side == side) & (summary_df.Route == route)].iloc[0].Dstar_m)
        c_route_rows.append({'Side': side, 'Route': route, 'C10': c10, 'Dstar_m': dstar, 'Unstable_after_10m': dstar > 10})
    c_route_df = pd.DataFrame(c_route_rows)
    c_route_df.to_csv(DATA / 'Qatraneh_R16_C10_RouteSummary.csv', index=False)
    for threshold in np.round(np.arange(0.80, 1.001, 0.01), 2):
        eligible = c_route_df[c_route_df['C10'] >= threshold - 1e-12]
        c_rows.append({
            'Threshold': float(threshold),
            'Routes_meeting_threshold_by_10m': int(len(eligible)),
            'Meeting_threshold_but_unstable_after_10m': int(eligible['Unstable_after_10m'].sum()),
        })
    pd.DataFrame(c_rows).to_csv(DATA / 'Qatraneh_R16_C10_ThresholdSensitivity.csv', index=False)

    # 4) Outer-tail analysis restricted to fully all-model-LOOCV-admissible windows.
    tail_rows = []
    for side, route, x_all, y_all in route_data(obs):
        full_model = summary_df[(summary_df.Side == side) & (summary_df.Route == route)].iloc[0].Full_model
        for w in TAIL_WINDOWS:
            inner_model = decisions_df[(decisions_df.Side == side) & (decisions_df.Route == route) & (decisions_df.Outer_window_m == w)].iloc[0].Model
            inner = x_all <= w
            tail = x_all > w
            tail_rmse = {}
            for model in ['G', 'P', 'E']:
                params = fit_model(model, x_all[inner], y_all[inner])
                pr = predict(model, params, x_all[tail])
                tail_rmse[model] = float(np.sqrt(np.mean((y_all[tail] - pr) ** 2)))
            tail_best = min(tail_rmse, key=lambda m: (tail_rmse[m], MODEL_ORDER[m]))
            tail_rows.append({
                'Side': side,
                'Route': route,
                'Outer_window_m': w,
                'Inner_LOOCV_winner': inner_model,
                'Complete_record_class': full_model,
                'Tail_best_model': tail_best,
                'Tail_RMSE_G': tail_rmse['G'],
                'Tail_RMSE_P': tail_rmse['P'],
                'Tail_RMSE_E': tail_rmse['E'],
                'Inner_disagrees_with_complete': inner_model != full_model,
            })
    tail_df = pd.DataFrame(tail_rows)
    tail_df.to_csv(DATA / 'Qatraneh_R16_OuterTail_FullyAdmissible.csv', index=False)
    disag = tail_df[tail_df['Inner_disagrees_with_complete']].copy()
    clean_counts = {
        'fully_admissible_disagreement_windows': int(len(disag)),
        'complete_record_class_tail_best': int((disag['Tail_best_model'] == disag['Complete_record_class']).sum()),
        'inner_winner_tail_best': int((disag['Tail_best_model'] == disag['Inner_LOOCV_winner']).sum()),
        'third_model_tail_best': int(((disag['Tail_best_model'] != disag['Complete_record_class']) & (disag['Tail_best_model'] != disag['Inner_LOOCV_winner'])).sum()),
    }

    # 5) Reconstruct the legacy 35-window aggregate transparently.
    # At 5 m, P is not LOOCV-admissible in every held-out fold, so the internal
    # model comparison is necessarily G vs E only. P is, however, fit-admissible
    # on the complete five-point inner record for distal-tail scoring.
    legacy_rows = []
    for side, route, x_all, y_all in route_data(obs):
        full_model = summary_df[(summary_df.Side == side) & (summary_df.Route == route)].iloc[0].Full_model
        inner5 = x_all <= 5
        scores5, winner5 = loocv_scores(x_all[inner5], y_all[inner5], models=('G', 'E'))
        tail5 = x_all > 5
        rm5 = {}
        for model in ['G', 'P', 'E']:
            params = fit_model(model, x_all[inner5], y_all[inner5])
            rm5[model] = float(np.sqrt(np.mean((y_all[tail5] - predict(model, params, x_all[tail5])) ** 2)))
        best5 = min(rm5, key=lambda m: (rm5[m], MODEL_ORDER[m]))
        legacy_rows.append({
            'Side': side, 'Route': route, 'Outer_window_m': 5,
            'Internal_candidate_set': 'G/E only',
            'Inner_LOOCV_winner': winner5,
            'Complete_record_class': full_model,
            'Tail_best_model': best5,
            'Tail_RMSE_G': rm5['G'], 'Tail_RMSE_P': rm5['P'], 'Tail_RMSE_E': rm5['E'],
        })
    for _, r in tail_df.iterrows():
        legacy_rows.append({
            'Side': r.Side, 'Route': r.Route, 'Outer_window_m': int(r.Outer_window_m),
            'Internal_candidate_set': 'G/P/E',
            'Inner_LOOCV_winner': r.Inner_LOOCV_winner,
            'Complete_record_class': r.Complete_record_class,
            'Tail_best_model': r.Tail_best_model,
            'Tail_RMSE_G': r.Tail_RMSE_G, 'Tail_RMSE_P': r.Tail_RMSE_P, 'Tail_RMSE_E': r.Tail_RMSE_E,
        })
    legacy_df = pd.DataFrame(legacy_rows)
    legacy_df['Inner_disagrees_with_complete'] = legacy_df['Inner_LOOCV_winner'] != legacy_df['Complete_record_class']
    legacy_df.to_csv(DATA / 'Qatraneh_R16_OuterTail_Legacy35_Reconstructed.csv', index=False)
    ld = legacy_df[legacy_df['Inner_disagrees_with_complete']]
    legacy_counts = {
        'legacy_disagreement_windows': int(len(ld)),
        'complete_record_class_tail_best': int((ld['Tail_best_model'] == ld['Complete_record_class']).sum()),
        'inner_winner_tail_best': int((ld['Tail_best_model'] == ld['Inner_LOOCV_winner']).sum()),
        'third_model_tail_best': int(((ld['Tail_best_model'] != ld['Complete_record_class']) & (ld['Tail_best_model'] != ld['Inner_LOOCV_winner'])).sum()),
    }

    # 6) Complete-profile model-class single-station jackknife.
    # This is distinct from LOOCV prediction: the entire model-selection procedure
    # is rerun after deleting each observed station from the complete 12-station route.
    jack_rows = []
    jack_summary = []
    for side, route, x_all, y_all in route_data(obs):
        full_scores, full_winner = loocv_scores(x_all, y_all)
        route_matches = 0
        changed_at = []
        for i, deleted_distance in enumerate(x_all):
            keep = np.arange(len(x_all)) != i
            scores_j, winner_j = loocv_scores(x_all[keep], y_all[keep])
            ordered_j = sorted(scores_j.items(), key=lambda kv: (kv[1], MODEL_ORDER[kv[0]]))
            margin_j = 100.0 * (ordered_j[1][1] - ordered_j[0][1]) / ordered_j[0][1]
            match = winner_j == full_winner
            route_matches += int(match)
            if not match:
                changed_at.append(float(deleted_distance))
            jack_rows.append({
                'Side': side,
                'Route': route,
                'Deleted_distance_m': float(deleted_distance),
                'Full_record_winner': full_winner,
                'Jackknife_winner': winner_j,
                'Matches_full_record': match,
                'Winner_margin_pct': margin_j,
                'LOOCV_RMSE_G': scores_j['G'],
                'LOOCV_RMSE_P': scores_j['P'],
                'LOOCV_RMSE_E': scores_j['E'],
            })
        jack_summary.append({
            'Side': side,
            'Route': route,
            'Full_record_winner': full_winner,
            'Matching_deletions_n': route_matches,
            'Total_deletions_n': int(len(x_all)),
            'Changed_deletion_distances_m': ';'.join(f'{d:g}' for d in changed_at) if changed_at else '',
        })
    jack_df = pd.DataFrame(jack_rows)
    jack_sum_df = pd.DataFrame(jack_summary)
    jack_df.to_csv(DATA / 'Qatraneh_R16_FullModel_SingleStationJackknife.csv', index=False)
    jack_sum_df.to_csv(DATA / 'Qatraneh_R16_FullModel_SingleStationJackknife_Summary.csv', index=False)

    # Key exact-margin checks already quoted in the manuscript.
    def get_margin(side, route, w):
        r = scores_df[(scores_df.Side == side) & (scores_df.Route == route) & (scores_df.Outer_window_m == w)].iloc[0]
        return float(r.Winner_margin_pct)

    exact_max = c_route_df[np.isclose(c_route_df['C10'], 1.0)]
    verification = {
        'lambda_grid_points': N_LAMBDA,
        'lambda_grid_spacing': 'logarithmic',
        'nested_decisions_recomputed': int(len(decisions_df)),
        'locked_decision_matches': match_count,
        'north_coarse_LF_full_margin_pct': get_margin('North', 'Coarse LF', 14),
        'north_coarse_HF_full_margin_pct': get_margin('North', 'Coarse HF', 14),
        'north_field_12m_margin_pct': get_margin('North', 'Field K', 12),
        'routes_C10_ge_0_90': int((c_route_df.C10 >= 0.90).sum()),
        'routes_C10_equal_1_00': int(len(exact_max)),
        'exact_max_routes_unstable_after_10m': int(exact_max['Unstable_after_10m'].sum()),
        'outer_tail_fully_admissible': clean_counts,
        'outer_tail_legacy35_reconstructed': legacy_counts,
        'legacy_third_model_cases': ld[(ld['Tail_best_model'] != ld['Complete_record_class']) & (ld['Tail_best_model'] != ld['Inner_LOOCV_winner'])][['Side', 'Route', 'Outer_window_m', 'Inner_LOOCV_winner', 'Complete_record_class', 'Tail_best_model']].to_dict('records'),
        'single_station_model_jackknife': {
            'matching_complete_model_decisions': int(jack_df['Matches_full_record'].sum()),
            'total_deletions': int(len(jack_df)),
            'routes_invariant_under_all_12_deletions': int((jack_sum_df['Matching_deletions_n'] == jack_sum_df['Total_deletions_n']).sum()),
            'all_12m_and_14m_deletions_match_complete_model': bool(jack_df[jack_df['Deleted_distance_m'].isin([12.0, 14.0])]['Matches_full_record'].all()),
            'changed_cases': jack_df.loc[~jack_df['Matches_full_record'], ['Side', 'Route', 'Deleted_distance_m', 'Full_record_winner', 'Jackknife_winner']].to_dict('records'),
        },
    }
    (DATA / 'Qatraneh_R16_Verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')

    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()

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
    """Fit E(d)=alpha+beta*d+A*exp(-d/lambda), A>=0, lambda>0."""
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
    sx = float(np.sum(x)); sx2 = float(x @ x); sy = float(np.sum(y)); sxy = float(x @ y)
    se = np.sum(e, axis=1); sxe = e @ x; see = np.sum(e * e, axis=1); sey = e @ y

    normal = np.empty((n_lambda, 3, 3), float)
    normal[:, 0, 0] = n; normal[:, 0, 1] = sx; normal[:, 1, 0] = sx; normal[:, 1, 1] = sx2
    normal[:, 0, 2] = se; normal[:, 2, 0] = se; normal[:, 1, 2] = sxe; normal[:, 2, 1] = sxe; normal[:, 2, 2] = see
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
    bad = coef[:, 2] < 0
    if np.any(bad):
        gcoef, gsse = fit_G(x, y)
        coef[bad, 0] = gcoef[0]; coef[bad, 1] = gcoef[1]; coef[bad, 2] = 0.0; sse[bad] = gsse
    j = int(np.argmin(sse))
    return (coef[j], float(lambdas[j])), float(sse[j])


def pred_E(params, x):
    coef, lam = params
    x = np.asarray(x, float)
    return coef[0] + coef[1] * x + coef[2] * np.exp(-x / lam)


def fit_model(model, x, y):
    if model == 'G': return fit_G(x, y)[0]
    if model == 'P': return fit_P(x, y)[0]
    if model == 'E': return fit_E(x, y)[0]
    raise KeyError(model)


def predict(model, params, x):
    if model == 'G': return pred_G(params, x)
    if model == 'P': return pred_P(params, x)
    if model == 'E': return pred_E(params, x)
    raise KeyError(model)


def loocv_scores(x, y, models=('G', 'P', 'E')):
    x = np.asarray(x, float); y = np.asarray(y, float)
    preds = {m: [] for m in models}
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        xt, yt = x[keep], y[keep]
        for model in models:
            params = fit_model(model, xt, yt)
            preds[model].append(float(predict(model, params, [x[i]])[0]))
    scores = {m: float(np.sqrt(np.mean((y - np.asarray(preds[m])) ** 2))) for m in models}
    winner = min(scores, key=lambda m: (scores[m], MODEL_ORDER[m]))
    return scores, winner


def dstar_from_sequence(seq_rows):
    full = seq_rows.iloc[-1]['Winner']
    wins = seq_rows['Winner'].tolist(); ws = seq_rows['Outer_window_m'].tolist()
    for i, w in enumerate(ws):
        if all(m == full for m in wins[i:]): return float(w)
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
    score_rows = []
    for side, route, x_all, y_all in route_data(obs):
        for w in WINDOWS:
            mask = x_all <= w
            scores, winner = loocv_scores(x_all[mask], y_all[mask])
            ordered = sorted(scores.items(), key=lambda kv: (kv[1], MODEL_ORDER[kv[0]]))
            margin_pct = 100.0 * (ordered[1][1] - ordered[0][1]) / ordered[0][1]
            score_rows.append({'Side':side,'Route':route,'Outer_window_m':w,'Winner':winner,'LOOCV_RMSE_G':scores['G'],'LOOCV_RMSE_P':scores['P'],'LOOCV_RMSE_E':scores['E'],'Runner_up':ordered[1][0],'Winner_margin_pct':margin_pct})
    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(DATA / 'Qatraneh_R16_GEP_LOOCV_Scores_RECOMPUTED.csv', index=False)

    decision_rows=[]; route_summary=[]
    for (side,route),q in scores_df.groupby(['Side','Route'],sort=False):
        q=q.sort_values('Outer_window_m'); full=q.iloc[-1]['Winner']; dstar=dstar_from_sequence(q)
        for _,r in q.iterrows(): decision_rows.append({'Side':side,'Route':route,'Outer_window_m':int(r['Outer_window_m']),'Model':r['Winner'],'Full_model':full,'Dstar_m':dstar})
        route_summary.append({'Side':side,'Route':route,'Full_model':full,'Dstar_m':dstar,'Full_winner_margin_pct':float(q.iloc[-1]['Winner_margin_pct'])})
    decisions_df=pd.DataFrame(decision_rows); decisions_df.to_csv(DATA/'Qatraneh_R16_NestedModelDecisions_RECOMPUTED.csv',index=False)
    summary_df=pd.DataFrame(route_summary); summary_df.to_csv(DATA/'Qatraneh_R16_ModelMarginSummary.csv',index=False)

    match_count=None
    if locked is not None:
        m=locked.merge(decisions_df,on=['Side','Route','Outer_window_m'],suffixes=('_locked','_recomputed'))
        m['Match']=m['Model_locked']==m['Model_recomputed']; m.to_csv(DATA/'Qatraneh_R16_Locked_vs_Recomputed.csv',index=False)
        match_count=int(m['Match'].sum())
        if match_count!=len(m): raise RuntimeError('Separate reimplementation mismatch')

    c_route_rows=[]
    for side,route,x_all,y_all in route_data(obs):
        c10=float(np.max(y_all[x_all<=10])/np.max(y_all)); dstar=float(summary_df[(summary_df.Side==side)&(summary_df.Route==route)].iloc[0].Dstar_m)
        c_route_rows.append({'Side':side,'Route':route,'C10':c10,'Dstar_m':dstar,'Unstable_after_10m':dstar>10})
    c_route_df=pd.DataFrame(c_route_rows); c_route_df.to_csv(DATA/'Qatraneh_R16_C10_RouteSummary.csv',index=False)
    rows=[]
    for threshold in np.round(np.arange(0.80,1.001,0.01),2):
        eligible=c_route_df[c_route_df['C10']>=threshold-1e-12]
        rows.append({'Threshold':float(threshold),'Routes_meeting_threshold_by_10m':int(len(eligible)),'Meeting_threshold_but_unstable_after_10m':int(eligible['Unstable_after_10m'].sum())})
    pd.DataFrame(rows).to_csv(DATA/'Qatraneh_R16_C10_ThresholdSensitivity.csv',index=False)

    print(json.dumps({'nested_decisions_recomputed':int(len(decisions_df)),'locked_decision_matches':match_count,'routes_C10_ge_0_90':int((c_route_df.C10>=0.90).sum()),'routes_C10_equal_1_00':int(np.isclose(c_route_df.C10,1.0).sum())},indent=2))


if __name__=='__main__':
    main()

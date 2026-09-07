from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'data'

spec = importlib.util.spec_from_file_location('gep', HERE / 'recompute_qatraneh_gep.py')
gep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gep)
obs = pd.read_csv(DATA / 'Qatraneh_R15R4_Observations.csv')
locked = pd.read_csv(DATA / 'Qatraneh_R15R4_NestedModelDecisions_LOCKED.csv')


def decisions_for_grid(n_lambda):
    def fit_model(model, x, y):
        if model == 'G':
            return gep.fit_G(x, y)[0]
        if model == 'P':
            return gep.fit_P(x, y)[0]
        if model == 'E':
            return gep.fit_E(x, y, n_lambda=n_lambda)[0]
        raise KeyError(model)

    def loocv_winner(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        preds = {m: [] for m in ('G', 'P', 'E')}
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            for model in preds:
                params = fit_model(model, x[keep], y[keep])
                preds[model].append(float(gep.predict(model, params, [x[i]])[0]))
        scores = {m: float(np.sqrt(np.mean((y - np.asarray(preds[m])) ** 2))) for m in preds}
        return min(scores, key=lambda m: (scores[m], gep.MODEL_ORDER[m]))

    rows = []
    for side, route, x, y in gep.route_data(obs):
        for w in gep.WINDOWS:
            mask = x <= w
            rows.append({'Side': side, 'Route': route, 'Outer_window_m': w,
                         'Winner': loocv_winner(x[mask], y[mask])})
    return pd.DataFrame(rows)


rows = []
for n in (51, 101, 251, 501, 1001):
    d = decisions_for_grid(n)
    m = locked.merge(d, on=['Side', 'Route', 'Outer_window_m'])
    rows.append({
        'Lambda_grid_points': n,
        'Matches_archived_decisions': int((m['Model'] == m['Winner']).sum()),
        'Total_decisions': int(len(m)),
        'All_decisions_identical': bool((m['Model'] == m['Winner']).all()),
    })

out = pd.DataFrame(rows)
out.to_csv(DATA / 'Qatraneh_R16_LambdaGridSensitivity.csv', index=False)
print(out.to_string(index=False))

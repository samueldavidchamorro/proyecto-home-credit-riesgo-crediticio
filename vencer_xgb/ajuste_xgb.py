"""TPE de 20 evaluaciones para XGBoost (GPU) en la partición de diseño; luego evalúa en la disjunta."""
import time
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from comun import cargar, particiones, guardar_json

optuna.logging.set_verbosity(optuna.logging.WARNING)
X, y = cargar()
P = particiones()


def entrenar(params, tr, va, n=5000):
    dtr = xgb.DMatrix(np.asarray(X[tr]), label=y[tr])
    dva = xgb.DMatrix(np.asarray(X[va]), label=y[va])
    p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda",
             seed=0, **params)
    b = xgb.train(p, dtr, n, evals=[(dva, "va")], early_stopping_rounds=100, verbose_eval=False)
    pred = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
    return roc_auc_score(y[va], pred), b.best_iteration + 1


def espacio(t):
    return dict(
        eta=t.suggest_float("eta", 0.01, 0.1, log=True),
        max_depth=t.suggest_int("max_depth", 3, 8),
        min_child_weight=t.suggest_float("min_child_weight", 1, 100, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-2, 100, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        gamma=t.suggest_float("gamma", 1e-3, 5, log=True),
        scale_pos_weight=t.suggest_float("scale_pos_weight", 1, 5),
    )


if __name__ == "__main__":
    t0 = time.time()
    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))

    def obj(t):
        a, it = entrenar(espacio(t), P["d_tr"], P["d_va"])
        t.set_user_attr("it", it)
        print(f"trial {t.number} auc {a:.5f} it {it}", flush=True)
        return a

    estudio.optimize(obj, n_trials=20)
    mejor = estudio.best_trial
    a_j, it_j = entrenar(mejor.params, P["j_tr"], P["j_va"])
    out = dict(params=mejor.params, it=mejor.user_attrs["it"], auc_diseno=mejor.value,
               auc_disjunta=a_j, it_disjunta=it_j, segundos=time.time() - t0)
    print(out)
    guardar_json("xgb_ajuste.json", out)

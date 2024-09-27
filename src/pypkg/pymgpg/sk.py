from pymgpg import conversion, complexity
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import sys, os
import inspect
import numpy as np
import sympy
import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pymgpg"
    ),
)

# load cpp interface
import _pb_mgpg


class MGPGRegressor(BaseEstimator, RegressorMixin):
    kwargs = {}

    def __init__(self, **kwargs):
        self.set_params(**kwargs)

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            self.kwargs[k] = v

    # def __del__(self):
    #  if hasattr(self, "_mgpg_cpp"):
    #    del self._mgpg_cpp

    def _create_cpp_option_string(self):
        # build string of options for cpp
        s = ""
        for k, v in self.kwargs.items():
            if v is None or isinstance(v, bool):
                if v:
                    s += f" -{k}"
            elif k == "max_time":
                # we let it run for 1 min less than it is set for SRBench only
                # to not have this, just use 't' directly instead of 'max_time'...
                s += f" -t {max(min(v, 60), v - 60)}"
            elif k in ["max_models"]:
                pass
            else:
                s += f" -{k} {v}"

        # add "lib" flag to differntiate from CLI calls
        s = s[1:] + " -lib"

        # init cpp interface object
        # & pass options for internal setup
        return s

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # setup cpp interface
        cpp_options = self._create_cpp_option_string()

        # impute if needed
        self.imputer = IterativeImputer(
            max_iter=10,
            random_state=self.kwargs.get("random_state", None),
            sample_posterior=True,
        )
        if np.isnan(X).any():
            X = self.imputer.fit_transform(X)
            # fix non-contiguous memory block for SWIG
            X = X.copy()
        else:
            self.imputer.fit(X)

        self.models = _pb_mgpg.evolve(cpp_options, X, y)
        # extract the model as a sympy and store it internally
        self.models, self.model = self._pick_best_models(X, y, self.models)

    def fit_val(self, X, y, X_val, y_val):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # setup cpp interface
        cpp_options = self._create_cpp_option_string()

        # impute if needed
        self.imputer = IterativeImputer(
            max_iter=10,
            random_state=self.kwargs.get("random_state", None),
            sample_posterior=True,
        )
        if np.isnan(X).any():
            X = self.imputer.fit_transform(X)
            # fix non-contiguous memory block for SWIG
            X = X.copy()
        else:
            self.imputer.fit(X)

        self.models = _pb_mgpg.evolve_val(cpp_options, X, y, X_val, y_val)
        # extract the model as a sympy and store it internally
        self.models, self.model = self._pick_best_models(X, y, self.models)

    def _finetune_multiple_models(self, models, X, y):
        import finetuning as ft

        if hasattr(self, "verbose") and self.verbose:
            print(f"finetuning {len(models)} models...")

        if len(X) > 10000:
            print(
                "[!] Warning: finetuning on large datasets (>10,000 obs.) can be slow, skipping..."
            )
        else:
            if hasattr(self, "finetune_max_evals"):
                # scatter finetuning over all models based on their number of coefficients
                num_coeffs = [complexity.get_num_coefficients(m) for m in models]
                tot = sum(num_coeffs)
                finetune_num_steps = [
                    int(self.finetune_max_evals * (n / tot)) for n in num_coeffs
                ]
                while sum(finetune_num_steps) < self.finetune_max_evals:
                    finetune_num_steps[np.random.randint(len(models))] += 1
            else:
                finetune_num_steps = [100] * len(models)

            for i, m in enumerate(models):
                models[i], steps_done = ft.finetune(
                    m, X, y, n_steps=finetune_num_steps[i]
                )
                steps_leftover = finetune_num_steps[i] - steps_done
                # scatter steps left over all models
                if i + 1 < len(models):
                    models_left = len(models) - i - 1
                    steps_left_per_model = int(steps_leftover / models_left)
                    reminder = steps_leftover % models_left
                    for j in range(i + 1, len(models)):
                        finetune_num_steps[j] += steps_left_per_model
                    # and scatter remainder
                    for j in range(reminder):
                        finetune_num_steps[np.random.randint(i + 1, len(models))] += 1

    def _pick_best_models(self, X, y, models):
        simplified_models = list()
        for m in models:
            simpl_m = conversion.timed_simplify(m, ratio=1.0, timeout=5)
            if simpl_m is None:
                simpl_m = sympy.sympify(m)  # do not simplify, just sympify
            simplified_models.append(simpl_m)
        # proceed with simplified models
        models = simplified_models

        # cleanup
        models = [conversion.model_cleanup(m, timeout=5) for m in models]
        models = [m for m in models if m is not None]

        # pick best
        errs = list()
        max_err = 0
        for i, m in enumerate(models):
            p = self.predict(X, model=m)
            if np.isnan(p).any():
                # convert this model to a constant, i.e., the mean over the training y
                models[i] = sympy.sympify(np.mean(y))
                p = np.array([np.mean(y)] * len(y))
            err = mean_squared_error(y, p)
            if err > max_err:
                max_err = err
            errs.append(err)
        # adjust errs
        errs = [err if not np.isnan(err) else max_err + 1e-6 for err in errs]
        by_err = np.argsort(errs)

        if hasattr(self, "rci") and len(models) > 1:
            complexity_metric = (
                "node_count" if not hasattr(self, "compl") else self.compl
            )
            compls = [
                complexity.compute_complexity(m, complexity_metric) for m in models
            ]
            best_idx = complexity.determine_rci_best(errs, compls, self.rci)
        else:
            best_idx = by_err[0]

        max_models = self.kwargs.get("max_models")
        return [
            models[i] for i in list(by_err)[:max_models]
        ] if max_models is not None else models, models[best_idx]

    def predict(self, X, model=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # impute if needed
        if np.isnan(X).any():
            assert hasattr(self, "imputer")
            X = self.imputer.transform(X)

        if model is None:
            assert self.model is not None
            # assume implicitly wanted the best one found at fit
            model = self.model

        # deal with a model that was simplified to a simple constant
        if isinstance(model, sympy.Float) or isinstance(model, sympy.Integer):
            prediction = np.array([float(model)] * X.shape[0])
            return prediction

        f = conversion.sympy_to_numpy_fn(model, timeout=5)
        if f is None:
            print(
                "[!] Warning: failed to convert sympy model to numpy, returning NaN as prediction"
            )
            return float("nan")

        try:
            prediction = f(X)
        except:
            print(
                "[!] Warning: failed to evaluate sympy model, returning NaN as prediction"
            )
            return float("nan")

        # can still happen for certain classes of sympy
        # (e.g., sympy.core.numbers.Zero)
        if isinstance(prediction, (int, float, np.int64, np.float64)):
            prediction = np.array([float(prediction)] * X.shape[0])
        if len(prediction) != X.shape[0]:
            prediction = np.array([prediction[0]] * X.shape[0])

        return prediction

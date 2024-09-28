import ast
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


def model_size(m) -> int:
    """Returns the model size"""

    # ! doesn't works since sympy internally uses + and * to represent e.g. -
    # sum(1 for _ in sym.preorder_traversal(self._model))

    s = 0
    q = [ast.parse(str(m)).body[0].value]
    while len(q) > 0:
        n = q.pop()
        if isinstance(n, (ast.Name, ast.Constant)):
            s += 1
        elif isinstance(n, ast.UnaryOp):
            s += 1
            q.append(n.operand)
        elif isinstance(n, ast.BinOp):
            s += 1
            q.append(n.left)
            q.append(n.right)
        elif isinstance(n, ast.Call):
            s += 1
            for arg in n.args:
                q.append(arg)

    return s


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
        def process_model(m):
            # simplify
            sm = conversion.timed_simplify(m, ratio=1.0, timeout=5)
            if sm is None:
                sm = sympy.sympify(m)  # do not simplify, just sympify

            # cleanup
            return conversion.model_cleanup(sm, timeout=5)

        # proceed with simplified & cleaned models
        models = [p for m in models if (p := process_model(m)) is not None]

        # compute error and size
        mse = np.empty(len(models))

        my = np.mean(y)
        for i, m in enumerate(models):
            p = self.predict(X, model=m)
            if np.isnan(p).any():
                # convert this model to a constant, i.e., the mean over the training y
                models[i] = sympy.sympify(my)
                p = np.repeat(my, y.shape[0])
            mse[i] = mean_squared_error(y, p)

        best_mse_idx = np.nanargmin(mse)

        # model selection
        max_models = self.kwargs.get("max_models")
        if max_models is None or max_models >= len(models):
            selected = models
        else:
            assert max_models > 0

            size = np.array([model_size(m) for m in models])

            selected_indices = [best_mse_idx]
            remaining_indices = set(range(len(models)))
            remaining_indices.remove(best_mse_idx)

            # greedy scattered subset selection: repeatedly add the most distant model
            # from the already selected using euclidean dist wrt mse and size
            while len(selected_indices) < max_models:
                best_dist = np.inf
                best_idx = None
                for i in remaining_indices:
                    dist = np.min(
                        np.abs(mse[selected_indices] - mse[i])
                        + np.abs(size[selected_indices] - size[i])
                    )
                    if dist > best_dist:
                        best_dist = dist
                        best_idx = i

                if best_idx is None:
                    break

                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

            selected = [models[i] for i in selected_indices]

        return selected, models[best_mse_idx]

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

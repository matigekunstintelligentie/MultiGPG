from pymgpg import conversion as C
import numpy as np
import sympy
from copy import deepcopy
import re


def complexity(m):
    c = 0
    for _ in sympy.preorder_traversal(m):
        c += 1
    return c


"""
Fine-tunes a sympy model. Returns the fine-tuned model and the number of steps used.
If it terminates prematurely, the number of steps used is returned as well.
"""


def finetune(
    sympy_model, X, y, learning_rate=1.0, n_steps=100, tol_grad=1e-9, tol_change=1e-9
):
    import torch

    best_torch_model, best_loss = None, np.infty

    if not isinstance(X, torch.TensorType):
        X = torch.tensor(X)
    if not isinstance(y, torch.TensorType):
        y = torch.tensor(y.reshape((-1,)))

    sympy_model_before = sympy_model

    sympy_model = sympy.sympify(str(sympy_model))

    if complexity(sympy_model_before) < complexity(sympy_model):
        sympy_model = sympy_model_before

    str_model = str(sympy_model).replace(" ", "")
    final_str_model = str_model

    # workaround to have identical constants be treated as different ones
    floats = []
    for el in sympy.preorder_traversal(sympy_model):
        if isinstance(el, sympy.Float):
            el = str(el)
            el = el.replace("-", "")
            f = float(el)
            floats.append(f + np.random.normal(0, 1e-5))
            str_model = str_model.replace(
                str(f), "floats[{}]".format(len(floats) - 1), 1
            )
            final_str_model = final_str_model.replace(
                str(f), "{" + str(len(floats) - 1) + "}"
            )

    pt_str = str_model.replace("cos", "torch.cos")
    pt_str = pt_str.replace("sin", "torch.sin")
    pt_str = pt_str.replace("sqrt", "torch.sqrt")
    pt_str = pt_str.replace("exp", "torch.exp")
    pt_str = pt_str.replace("log", "torch.log")
    pt_str = pt_str.replace("exp", "torch.exp")
    pt_str = pt_str.replace("min", "torch.min")
    pt_str = pt_str.replace("max", "torch.max")

    expr_vars = sorted(
        [int(el.replace("x_", "")) for el in set(re.findall(r"\bx_[0-9]+", pt_str))],
        reverse=True,
    )

    v = [X[:, [i]] for i in range(X.size(1))]

    for el in expr_vars:
        pt_str = pt_str.replace("x_{}".format(el), "v[{}]".format(el))

    floats = [torch.tensor(f, requires_grad=True) for f in floats]

    try:  # optimizer might get an empty parameter list
        optimizer = torch.optim.LBFGS(
            floats,
            line_search_fn=None,
            lr=learning_rate,
            tolerance_grad=tol_grad,
            tolerance_change=tol_change,
        )
    except ValueError:
        return sympy_model, 0

    prev_loss = np.infty
    steps_done = 0
    for _ in range(n_steps):
        steps_done += 1
        optimizer.zero_grad()
        try:
            p = eval(pt_str).squeeze(-1)
        except TypeError:
            print(
                "[!] Warning: error during forward call of torch model while fine-tuning"
            )
            return sympy_model, steps_done

        loss = (p - y).pow(2).mean().div(2)

        loss.retain_grad()
        loss.backward()
        optimizer.step(lambda: loss)
        loss_val = loss.item()
        if loss_val < best_loss:
            best_torch_model = deepcopy(floats)
            best_loss = loss_val
        if abs(loss_val - prev_loss) < tol_change:
            break
        prev_loss = loss_val

    # =============================================================================
    #     if steps_done==1:
    #       print(1, prev_loss*2, [el.detach().item() for el in floats])
    #
    #   print(2, prev_loss*2,[el.detach().item() for el in floats])
    #   print(3,final_str_model)
    #   print(4, str(final_str_model.format(*[el.detach().item() for el in floats])))
    # =============================================================================

    result = (
        sympy.sympify(
            str(
                final_str_model.format(*[el.detach().item() for el in best_torch_model])
            )
        )
        if best_torch_model
        else sympy_model
    )

    result = C.timed_simplify(result, timeout=5)

    if result is None:
        return sympy_model, steps_done
    return result, steps_done

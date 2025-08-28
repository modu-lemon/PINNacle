import argparse
import itertools
import json
import os
import time
from copy import deepcopy

import numpy as np
import deepxde as dde

from trainer import Trainer

from src.utils.args import parse_hidden_layers
from src.utils.summary import summary as summarize

from src.pde.burgers import Burgers1D, Burgers2D
from src.pde.poisson import Poisson2D_Classic
from src.pde.heat import Heat2D_VaryingCoef
from src.pde.wave import Wave1D


ALL_PDES = [
    Burgers1D,
    Burgers2D,
    Poisson2D_Classic,
    Heat2D_VaryingCoef,
    Wave1D,
]

ALL_EXPERTS = ["fnn", "laaf", "gaaf"]


def parse_combinations(experts, max_k=None):
    names = [e.strip() for e in experts.split(',') if e.strip()]
    if not names:
        names = ALL_EXPERTS
    combos = []
    for r in range(1, len(names) + 1):
        if max_k is not None and r > max_k:
            break
        combos.extend(itertools.combinations(names, r))
    return [list(c) for c in combos]


def make_get_model(pde_ctor, expert_list, hidden_layers, gating_hidden, top_k, method, lr, loss_weights=None):
    import numpy as np
    import torch
    from src.model.moe import build_moe
    from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS
    from src.utils.rar import rar_wrapper
    from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback

    def fn():
        pde = pde_ctor()
        if method == "gepinn":
            pde.use_gepinn()

        net = build_moe(expert_list, pde, hidden_layers, gating_hidden=gating_hidden, top_k=top_k)
        net = net.float()

        if loss_weights is None:
            lw = np.ones(pde.num_loss)
        else:
            lw = np.array(loss_weights)

        opt = torch.optim.Adam(net.parameters(), lr)
        if method == "multiadam":
            opt = MultiAdam(net.parameters(), lr=1e-3, betas=(0.99, 0.99), loss_group_idx=[pde.num_pde])
        elif method == "lra":
            opt = LR_Adaptor(opt, lw, pde.num_pde)
        elif method == "ntk":
            opt = LR_Adaptor_NTK(opt, lw, pde)
        elif method == "lbfgs":
            opt = Adam_LBFGS(net.parameters(), switch_epoch=5000, adam_param={'lr':lr})

        model = pde.create_model(net)
        model.compile(opt, loss_weights=lw)
        if method == "rar":
            model.train = rar_wrapper(pde, model, {"interval": 1000, "count": 1})
        return model

    return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search best MoE expert combination')
    parser.add_argument('--name', type=str, default="moe-search")
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hidden-layers', type=str, default="100*5")
    parser.add_argument('--gating-hidden', type=str, default="64*2")
    parser.add_argument('--experts', type=str, default="fnn,laaf,gaaf")
    parser.add_argument('--max-experts', type=int, default=None)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--pdes', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iter', type=int, default=20000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--plot-every', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--method', type=str, default="adam")

    args = parser.parse_args()

    if args.seed is not None:
        dde.config.set_random_seed(args.seed)

    class _Args: pass
    _a = _Args(); _a.hidden_layers = args.gating_hidden
    gating_hidden = parse_hidden_layers(_a)
    _b = _Args(); _b.hidden_layers = args.hidden_layers
    hidden_layers = parse_hidden_layers(_b)

    if args.pdes:
        # Map by class name substring
        selected = []
        tokens = [s.strip().lower() for s in args.pdes.split(',') if s.strip()]
        for p in ALL_PDES:
            if any(tok in p.__name__.lower() for tok in tokens):
                selected.append(p)
        pdes = selected or ALL_PDES
    else:
        pdes = ALL_PDES

    combos = parse_combinations(args.experts, args.max_experts)

    date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
    trainer = Trainer(f"{date_str}-{args.name}", args.device)

    tasks_meta = []
    for pde_ctor in pdes:
        for expert_list in combos:
            get_model = make_get_model(
                pde_ctor=pde_ctor,
                expert_list=expert_list,
                hidden_layers=hidden_layers,
                gating_hidden=gating_hidden,
                top_k=args.top_k,
                method=args.method,
                lr=args.lr,
            )
            trainer.add_task(
                get_model,
                {
                    "iterations": args.iter,
                    "display_every": args.log_every,
                    "callbacks": []  # keep it lean for search; logs will still be written
                }
            )
            tasks_meta.append({"pde": pde_ctor.__name__, "experts": expert_list})

    trainer.setup(__file__, args.seed)
    trainer.set_repeat(args.repeat)
    trainer.train_all()

    # Aggregate results and select best per PDE based on final MSE
    # Reuse existing summary csv and then post-filter
    trainer.summary()

    # Additionally, produce a lightweight JSON per PDE with best combo by mse
    result_dir = f"runs/{trainer.exp_name}"
    # Build in-memory index of task -> meta
    best = {}
    for idx, meta in enumerate(tasks_meta):
        pde_name = meta["pde"]
        try:
            arr = np.loadtxt(f"{result_dir}/{idx}-0/errors.txt")
            mse = float(arr[-1, 2])
        except Exception:
            mse = float("inf")
        if pde_name not in best or mse < best[pde_name]["mse"]:
            best[pde_name] = {"mse": mse, "experts": meta["experts"], "task_index": idx}

    with open(f"{result_dir}/best_by_pde.json", "w") as f:
        json.dump(best, f, indent=2)

    print("Best combos by PDE saved to:", f"{result_dir}/best_by_pde.json")


#!/usr/bin/env python3

import os
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

from wremnants.utilities.io_tools import base_io


REGIONS = {
    "A_phys": {"dxy": 1, "relIso": 1},
    "B_phys": {"dxy": 1, "relIso": 0},
    "C_phys": {"dxy": 0, "relIso": 1},
    "D_phys": {"dxy": 0, "relIso": 0},
}


MODEL_TO_PHYS = {
    "A_model": "B_phys",
    "B_model": "A_phys",
    "C_model": "C_phys",
}


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--inputFile", required=True)
    p.add_argument("--histName", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("-o", "--outdir", required=True)
    p.add_argument("--smoothingAxis", required=True)
    p.add_argument("--dxyAxis", default="dxy")
    p.add_argument("--relIsoAxis", default="relIso")
    p.add_argument("--dataProcFilter", default="SingleMuon")
    p.add_argument(
        "--promptProcFilters",
        nargs="+",
        default=["Wplusmunu", "Wminusmunu", "Wplustaunu", "Wminustaunu", "Zmumu", "Ztautau"],
    )
    p.add_argument("--fakeFloor", type=float, default=1e-6)

    # Choose a representative outer bin.
    # Example for mu_pt: --outerSelect eta:24 charge:0
    # Example for w_pt with w_mt: --outerSelect eta:24 charge:0 w_mt:1
    p.add_argument("--outerSelect", nargs="*", default=[])

    return p


def get_hist(results, proc, hist_name):
    h_proxy = results[proc]["output"][hist_name]
    return h_proxy.get() if hasattr(h_proxy, "get") else h_proxy


def has_hist(results, proc, hist_name):
    return proc in results and "output" in results[proc] and hist_name in results[proc]["output"]


def find_one_process(processes, filt):
    matches = [p for p in processes if filt in p]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one process matching '{filt}', found {matches}")
    return matches[0]


def build_fake_seed(input_file, hist_name, data_filter, prompt_filters, fake_floor):
    with h5py.File(input_file, "r") as f:
        results = base_io.load_results_h5py(f)

        all_procs = [p for p in results.keys() if p != "meta_info"]
        data_proc = find_one_process(all_procs, data_filter)

        h_fake = get_hist(results, data_proc, hist_name).copy()
        v_fake = h_fake.view(flow=False)

        subtracted = []

        for proc in all_procs:
            if proc == data_proc:
                continue

            if not any(filt in proc for filt in prompt_filters):
                continue

            if not has_hist(results, proc, hist_name):
                continue

            h_prompt = get_hist(results, proc, hist_name)

            if h_prompt.axes.name != h_fake.axes.name:
                raise RuntimeError(
                    f"Axis mismatch for {proc}: "
                    f"data axes={h_fake.axes.name}, prompt axes={h_prompt.axes.name}"
                )

            v_fake.value[...] -= h_prompt.view(flow=False).value
            subtracted.append(proc)

        before_floor = v_fake.value.copy()
        floor_mask = v_fake.value <= fake_floor
        v_fake.value[floor_mask] = fake_floor

        return h_fake, before_floor, floor_mask, subtracted

    return h_fake, before_floor, floor_mask, subtracted


def make_chebyshev_matrix(axis, order):
    centers = np.array(axis.centers)
    edges = np.array(axis.edges)

    x_tilde = 2.0 * (centers - edges[0]) / (edges[-1] - edges[0]) - 1.0

    T = np.zeros((len(x_tilde), order + 1))
    T[:, 0] = 1.0

    if order >= 1:
        T[:, 1] = x_tilde

    for k in range(2, order + 1):
        T[:, k] = 2.0 * x_tilde * T[:, k - 1] - T[:, k - 2]

    return centers, T


def parse_outer_select(items):
    out = {}
    for item in items:
        name, val = item.split(":", 1)
        out[name] = int(val)
    return out


def get_outer_axes(axes, smoothing_axis, dxy_axis, reliso_axis):
    return [ax for ax in axes if ax not in [smoothing_axis, dxy_axis, reliso_axis]]


def get_outer_bins(outer_axes, outer_shape, outer_select):
    bins = []

    for ax, nbin in zip(outer_axes, outer_shape):
        if ax in outer_select:
            b = outer_select[ax]
        else:
            b = nbin // 2

        if b < 0 or b >= nbin:
            raise RuntimeError(f"Outer bin {ax}:{b} outside valid range [0,{nbin-1}]")

        bins.append(b)

    return tuple(bins)


def get_region(values, axes, smoothing_axis, outer_axes, outer_bins, dxy_axis, reliso_axis, region):
    slicer = [slice(None)] * len(axes)

    for ax, b in zip(outer_axes, outer_bins):
        slicer[axes.index(ax)] = b

    slicer[axes.index(dxy_axis)] = REGIONS[region]["dxy"]
    slicer[axes.index(reliso_axis)] = REGIONS[region]["relIso"]

    y = values[tuple(slicer)]

    if y.ndim != 1:
        raise RuntimeError(f"Expected 1D region slice, got shape {y.shape}")

    return y


def load_params(params_file):
    with h5py.File(params_file, "r") as f:
        params = f["params"][()]
        order = int(f["order"][()])
        n_outer = int(f["n_outer"][()])
        smoothing_axis = f["smoothing_axis_name"][()].decode()
        outer_shape = tuple(f["outer_shape"][()])

    if not np.isfinite(params).all():
        raise RuntimeError(
            f"Params contain non-finite values: nan={np.isnan(params).sum()}, inf={np.isinf(params).sum()}"
        )

    expected = 3 * n_outer * (order + 1)
    if params.shape[0] != expected:
        raise RuntimeError(f"Params length mismatch: got {params.shape[0]}, expected {expected}")

    q = params.reshape(3, n_outer, order + 1)

    return q, order, n_outer, smoothing_axis, outer_shape


def plot_region(outdir, x, target, reco, title, fname):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(x, target, where="mid", label="fake seed: data - prompt MC")
    ax.plot(x, reco, marker="o", linestyle="-", label="Chebyshev reconstruction")
    ax.set_xlabel("smoothing axis")
    ax.set_ylabel("yield / bin")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ratio = reco / np.maximum(target, 1e-12)
    ax.axhline(1.0, linestyle="--")
    ax.plot(x, ratio, marker="o", linestyle="-")
    ax.set_xlabel("smoothing axis")
    ax.set_ylabel("Chebyshev / target")
    ax.set_ylim(0.0, 2.0)
    ax.set_title(title + " ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname.replace(".png", "_ratio.png")))
    plt.close(fig)


def main():
    args = make_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    h_fake, before_floor, floor_mask, subtracted = build_fake_seed(
        args.inputFile,
        args.histName,
        args.dataProcFilter,
        args.promptProcFilters,
        args.fakeFloor,
    )

    axes = list(h_fake.axes.name)
    values = h_fake.view(flow=False).value

    q, order, n_outer, params_smoothing_axis, outer_shape = load_params(args.params)

    if params_smoothing_axis != args.smoothingAxis:
        raise RuntimeError(
            f"Params smoothing axis is {params_smoothing_axis}, but requested {args.smoothingAxis}"
        )

    outer_axes = get_outer_axes(axes, args.smoothingAxis, args.dxyAxis, args.relIsoAxis)
    hist_outer_shape = tuple(len(h_fake.axes[ax]) for ax in outer_axes)

    if hist_outer_shape != outer_shape:
        raise RuntimeError(
            f"Outer shape mismatch: hist has {hist_outer_shape} for {outer_axes}, params has {outer_shape}"
        )

    outer_select = parse_outer_select(args.outerSelect)
    outer_bins = get_outer_bins(outer_axes, outer_shape, outer_select)
    outer_flat = np.ravel_multi_index(outer_bins, outer_shape)

    x, T = make_chebyshev_matrix(h_fake.axes[args.smoothingAxis], order)

    print("\n=== Inputs ===")
    print(f"hist file: {args.inputFile}")
    print(f"hist name: {args.histName}")
    print(f"params file: {args.params}")
    print(f"subtracted prompt processes: {subtracted}")
    print(f"hist axes: {axes}")
    print(f"smoothing axis: {args.smoothingAxis}")
    print(f"outer axes: {outer_axes}")
    print(f"outer shape: {outer_shape}")
    print(f"chosen outer bins: {dict(zip(outer_axes, outer_bins))}")
    print(f"outer flat index: {outer_flat}")
    print(f"order: {order}")
    print(f"T shape: {T.shape}")
    print(f"params shape: {q.shape}")
    print(f"fake seed finite: {np.isfinite(values).all()}")
    print(f"fake seed min/max/sum after floor: {values.min()} / {values.max()} / {values.sum()}")
    print(f"bins floored: {np.count_nonzero(floor_mask)}")

    model_regions = [
        ("A_model", 0),
        ("B_model", 1),
        ("C_model", 2),
    ]

    print("\n=== Coefficients for chosen outer bin ===")
    for model_name, idx in model_regions:
        phys_name = MODEL_TO_PHYS[model_name]
        coeff = q[idx, outer_flat, :]
        print(f"{model_name} <- {phys_name}: {coeff}")

    print("\n=== Region reconstruction ===")
    reconstructed = {}

    for model_name, idx in model_regions:
        phys_name = MODEL_TO_PHYS[model_name]
        target = get_region(
            values, axes, args.smoothingAxis, outer_axes, outer_bins,
            args.dxyAxis, args.relIsoAxis, phys_name,
        )
        pred = np.exp(T @ q[idx, outer_flat, :])
        reconstructed[model_name] = pred

        print(f"{model_name} <- {phys_name}")
        print(f"  target integral: {target.sum()}")
        print(f"  cheb integral:   {pred.sum()}")
        print(f"  ratio:           {pred.sum() / target.sum()}")

        plot_region(
            args.outdir,
            x,
            target,
            pred,
            f"{model_name} from {phys_name}, outer={dict(zip(outer_axes, outer_bins))}",
            f"{model_name}_{phys_name}.png",
        )

    # Predicted signal-like region from SmoothABCD:
    # D = A_model * C_model / B_model = B_phys * C_phys / A_phys
    D_seed = get_region(
        values, axes, args.smoothingAxis, outer_axes, outer_bins,
        args.dxyAxis, args.relIsoAxis, "D_phys",
    )

    D_pred = reconstructed["A_model"] * reconstructed["C_model"] / np.maximum(
        reconstructed["B_model"], 1e-12
    )

    print("\n=== ABCD-predicted D region ===")
    print(f"D seed integral: {D_seed.sum()}")
    print(f"D ABCD pred integral: {D_pred.sum()}")
    print(f"D pred / D seed: {D_pred.sum() / D_seed.sum()}")

    plot_region(
        args.outdir,
        x,
        D_seed,
        D_pred,
        f"D_phys seed compared to ABCD prediction, outer={dict(zip(outer_axes, outer_bins))}",
        "D_phys_abcd_prediction.png",
    )

    print(f"\nSaved plots in: {args.outdir}")


if __name__ == "__main__":
    main()
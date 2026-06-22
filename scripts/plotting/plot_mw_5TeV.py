import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep

from wremnants.utilities.io_tools import input_tools
from wums import plot_tools

import hist
import numpy as np

MC_GROUPS = {
    r"$Z/\gamma^*\rightarrow\mu\mu$": ["Zmumu_2017G", "Zmumu"],
    r"$Z/\gamma^*\rightarrow\tau\tau$": ["Ztautau_2017G", "Ztautau"],
    r"$W\rightarrow\tau\nu$": ["Wplustaunu_2017G","Wminustaunu_2017G", "Wtaunu"],
    r"$W\rightarrow\mu\nu$": ["Wplusmunu_2017G", "Wminusmunu_2017G", "Wmunu"],
}


MC_COLORS = [
    "#7DB7E8",  # blue
    "#F4A259",  # orange
    "#7BC67B",  # green
    "#C44E52",  # red
]

def load_results(infile):
    obj = input_tools.read_infile(infile)
    return obj[0] if isinstance(obj, tuple) else obj

def read_hist(results, sample, histname):
    return results[sample]["output"][histname].get()

def samples_in_file(results):
    return [k for k in results.keys() if k != "meta_info"]

def get_data_hist(results, histname):
    data_samples = [s for s in samples_in_file(results) if s.startswith("SingleMuon")]
    if len(data_samples) != 1:
        raise RuntimeError(f"Expected one SingleMuon sample, found {data_samples}")

    h = read_hist(results, data_samples[0], histname)
    y = h.sum().value if hasattr(h.sum(), "value") else h.sum()
    print(f"{data_samples[0]:25s} {histname:15s} yield = {y}")
    return h

def sum_hists(results, samples, histname):
    hsum = None

    for sample in samples:
        if sample not in results:
            print(f"Skipping missing sample: {sample}")
            continue

        h = read_hist(results, sample, histname)
        y = h.sum().value if hasattr(h.sum(), "value") else h.sum()
        print(f"{sample:25s} {histname:15s} yield = {y}")

        hsum = h.copy() if hsum is None else hsum + h

    return hsum

def get_mc_stack(results, histname):
    hists = []
    labels = []

    for label, samples in MC_GROUPS.items():
        h = sum_hists(results, samples, histname)
        if h is None:
            continue

        hists.append(h)
        labels.append(label)

    if not hists:
        raise RuntimeError(f"No MC histograms found for {histname}")
    return hists, labels


def make_plot(infile, outdir, histname, xlabel, xlim, lumi, com, logy):
    os.makedirs(outdir, exist_ok=True)

    results = load_results(infile)
    print("Available samples:", samples_in_file(results))

    mc_hists, mc_labels = get_mc_stack(results, histname)
    data_hist = get_data_hist(results, histname)

    fig, ax = plt.subplots(figsize=(10.5, 8.0))

    colors = MC_COLORS[:len(mc_hists)]

    hep.histplot(
        mc_hists,
        ax=ax,
        stack=True,
        histtype="fill",
        color=colors[:len(mc_hists)],
        alpha=0.75,
        linewidth=0,
        label=mc_labels,
        flow="none",
    )

    hep.histplot(
        mc_hists,
        ax=ax,
        stack=True,
        histtype="step",
        color="black",
        linewidth=1.0,
        flow="none",
    )

    hep.histplot(
        data_hist,
        ax=ax,
        histtype="errorbar",
        color="black",
        label="Data",
        flow="none",
    )

    if histname == "met_pt":
        for cut in [20, 25, 30, 35, 40]:
            ax.axvline(cut, color="black", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.axvline(25, color="black", linewidth=1.5, label=r"$p_T^{miss}>25$ GeV")

    if histname == "w_mt":
        ax.axvline(40, color="black", linewidth=1.5, label=r"$m_T>40$ GeV")

    ax.set_xlabel(xlabel if xlabel else histname, fontsize=14)
    ax.set_ylabel("Events/bin", fontsize=14)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if logy:
        ax.set_yscale("log")

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.minorticks_on()

    plot_tools.addLegend(ax, ncols=1, loc="upper right", text_size="small")

    hep.cms.label(
        ax=ax,
        label="Preliminary",
        data=True,
        lumi=lumi,
        com=com,
        loc=0,
    )

    fig.tight_layout()

    basename = f"{histname}_nominal"
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)

def slice_alphaS(h, variation):
    return h[{"alphaS": hist.loc(variation)}]

def make_ratio_vals(h_var, h_central):
    var_vals = h_var.values(flow=False)
    cen_vals = h_central.values(flow=False)

    return np.divide(
        var_vals,
        cen_vals,
        out=np.ones_like(var_vals, dtype=float),
        where=cen_vals != 0,
    )

def sum_alphaS_hists(results, samples, histname):
    h_central_sum = None
    h_up_sum = None
    h_down_sum = None

    for sample in samples:
        if sample not in results:
            print(f"Skipping missing sample: {sample}")
            continue

        if histname not in results[sample]["output"]:
            print(f"Skipping missing hist {histname} for sample {sample}")
            continue

        h = read_hist(results, sample, histname)

        h_central = slice_alphaS(h, "central")
        h_up = slice_alphaS(h, "alphaSUp")
        h_down = slice_alphaS(h, "alphaSDown")

        y_central = h_central.sum().value if hasattr(h_central.sum(), "value") else h_central.sum()
        y_up = h_up.sum().value if hasattr(h_up.sum(), "value") else h_up.sum()
        y_down = h_down.sum().value if hasattr(h_down.sum(), "value") else h_down.sum()

        print(f"{sample:25s} {histname:20s} central    yield = {y_central}")
        print(f"{sample:25s} {histname:20s} alphaSUp   yield = {y_up}")
        print(f"{sample:25s} {histname:20s} alphaSDown yield = {y_down}")

        h_central_sum = h_central.copy() if h_central_sum is None else h_central_sum + h_central
        h_up_sum = h_up.copy() if h_up_sum is None else h_up_sum + h_up
        h_down_sum = h_down.copy() if h_down_sum is None else h_down_sum + h_down

    if h_central_sum is None:
        raise RuntimeError(f"No alphaS histograms found for {histname}")

    return h_central_sum, h_up_sum, h_down_sum

def make_alphaS_plot(infile, outdir, histname, samples, xlabel, xlim, lumi, com, logy):
    os.makedirs(outdir, exist_ok=True)

    results = load_results(infile)
    print("Available samples:", samples_in_file(results))
    print("Using alphaS samples:", samples)

    h_central, h_up, h_down = sum_alphaS_hists(results, samples, histname)

    # This plotting function is for 1D hists like w_pt_alphaS and w_mt_alphaS.
    # For 2D hists like wpt_y_plus_alphaS, use an unrolled plotting script.
    if len(h_central.axes) != 1:
        raise RuntimeError(
            f"{histname} has {len(h_central.axes)} axes after alphaS slicing. "
            "Use an unrolled alphaS plotting script for 2D histograms."
        )

    ratio_up = make_ratio_vals(h_up, h_central)
    ratio_down = make_ratio_vals(h_down, h_central)

    x = h_central.axes[0].centers

    fig, (ax, rax) = plt.subplots(2,1,figsize=(10.5, 9.0),
        sharex=True,gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    hep.histplot(
        h_central,
        ax=ax,
        histtype="step",
        linewidth=1.8,
        label="central",
        flow="none",
    )

    hep.histplot(
        h_up,
        ax=ax,
        histtype="step",
        linewidth=1.5,
        label=r"$\alpha_s$ up",
        flow="none",
    )

    hep.histplot(
        h_down,
        ax=ax,
        histtype="step",
        linewidth=1.5,
        label=r"$\alpha_s$ down",
        flow="none",
    )

    ax.set_xlabel("")
    ax.set_ylabel("Events/bin", fontsize=14)

    if logy:
        ax.set_yscale("log")

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.minorticks_on()

    plot_tools.addLegend(ax, ncols=1, loc="upper right", text_size="small")

    hep.cms.label(
        ax=ax,
        label="Preliminary",
        data=True,
        lumi=lumi,
        com=com,
        loc=0,
    )

    rax.axhline(1.0, color="black", linewidth=1.0)
    rax.plot(x, ratio_up, marker="o", linestyle="none", markersize=3, label=r"$\alpha_s$ up / central")
    rax.plot(x, ratio_down, marker="o", linestyle="none", markersize=3, label=r"$\alpha_s$ down / central")

    rax.set_xlabel(xlabel if xlabel else histname, fontsize=14)
    rax.set_ylabel("Var./central", fontsize=12)
    rax.set_ylim(0.95, 1.05)

    if xlim is not None:
        rax.set_xlim(*xlim)

    rax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    rax.minorticks_on()

    fig.tight_layout()

    # sample_tag = "_".join(samples).replace("/", "")
    basename = f"{histname}"
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="HDF5 output from mw_5TeV.py")
    parser.add_argument("--hist", required=True, help="Histogram name, e.g. met_pt, w_mt, mu_pt")
    parser.add_argument("-o", "--outdir", default="plots_mw_5TeV")
    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--lumi", type=float, default=0.300)
    parser.add_argument("--com", default="5.02")
    parser.add_argument("--mode",
                        choices=["dataMC", "alphaS"],default="dataMC",
                        help="dataMC for nominal data/MC stack, alphaS for alphaS variation templates")
    parser.add_argument("--samples",nargs="*",default=None,help="MC samples to use for alphaS plotting")

    args = parser.parse_args()

    if args.mode == "dataMC":
        make_plot(
            args.infile,
            args.outdir,
            args.hist,
            args.xlabel,
            args.xlim,
            args.lumi,
            args.com,
            args.logy,
        )

    elif args.mode == "alphaS":
        samples_for_alphaS = args.samples

        if samples_for_alphaS is None or len(samples_for_alphaS) == 0:
            samples_for_alphaS = [
                "Wplusmunu_2017G",
                "Wminusmunu_2017G",
                "Wmunu",
            ]

        make_alphaS_plot(
            args.infile,
            args.outdir,
            args.hist,
            samples_for_alphaS,
            args.xlabel,
            args.xlim,
            args.lumi,
            args.com,
            args.logy,
        )

if __name__ == "__main__":
    main()
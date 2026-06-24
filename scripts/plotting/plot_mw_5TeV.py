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

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MC_GROUPS = {
    r"$Z/\gamma^*\rightarrow\mu\mu$": ["Zmumu_2017G", "Zmumu"],
    r"$Z/\gamma^*\rightarrow\tau\tau$": ["Ztautau_2017G", "Ztautau"],
    r"$W\rightarrow\tau\nu$": ["Wplustaunu_2017G", "Wminustaunu_2017G", "Wtaunu"],
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


def resolve_variation_histname(histname, variation):
    suffix = f"_{variation}"

    if histname.endswith(suffix):
        return histname

    return f"{histname}{suffix}"


def variation_axis_name(variation):
    if variation == "alphaS":
        return "alphaS"

    if variation == "pdf":
        return "pdf"

    raise RuntimeError(f"Unknown variation type: {variation}")


def variation_axis_labels(h, axis_name):
    axis = None

    for ax in h.axes:
        if ax.name == axis_name:
            axis = ax
            break

    if axis is None:
        raise RuntimeError(f"Could not find axis '{axis_name}' in histogram with axes {[ax.name for ax in h.axes]}")

    try:
        return [axis.value(i) for i in range(axis.size)]
    except Exception:
        return list(axis)


def slice_variation(h, axis_name, label):
    return h[{axis_name: hist.loc(label)}]


def hist_integral(h):
    s = h.sum()
    return s.value if hasattr(s, "value") else s


def scale_hist_to_ref_integral(h, h_ref):
    h_sum = hist_integral(h)
    ref_sum = hist_integral(h_ref)

    if h_sum == 0:
        return h.copy()

    return h * (ref_sum / h_sum)


def sum_variation_hists(results, samples, histname, axis_name):
    labels = None
    sums = {}

    for sample in samples:
        if sample not in results:
            print(f"Skipping missing sample: {sample}")
            continue

        if histname not in results[sample]["output"]:
            print(f"Skipping missing hist {histname} for sample {sample}")
            continue

        h = read_hist(results, sample, histname)
        sample_labels = variation_axis_labels(h, axis_name)

        if labels is None:
            labels = sample_labels
            sums = {label: None for label in labels}
        elif sample_labels != labels:
            raise RuntimeError(
                f"Variation labels changed for sample {sample}. "
                f"Expected {labels}, got {sample_labels}"
            )

        print(f"{sample:25s} {histname:20s} {axis_name} variations = {len(labels)}")

        for label in labels:
            h_var = slice_variation(h, axis_name, label)
            sums[label] = h_var.copy() if sums[label] is None else sums[label] + h_var

    if labels is None:
        raise RuntimeError(f"No variation histograms found for {histname}")

    central_label = labels[0]
    print(f"Central variation label: {central_label}")
    print(f"Central integral: {hist_integral(sums[central_label])}")

    if len(labels) <= 6:
        for label in labels:
            print(f"  {label:20s} integral = {hist_integral(sums[label])}")
    else:
        print(f"Number of non-central variations: {len(labels) - 1}")
        print(f"First few labels: {labels[:6]}")

    return labels, sums


def make_ratio_values(h_var, h_central, shape_norm=True):
    h_use = scale_hist_to_ref_integral(h_var, h_central) if shape_norm else h_var

    num = np.asarray(h_use.values(flow=False), dtype=float)
    den = np.asarray(h_central.values(flow=False), dtype=float)

    return np.divide(
        num,
        den,
        out=np.full_like(den, np.nan, dtype=float),
        where=den != 0,
    )


def make_alphaS_plot(h_central, h_up, h_down, histname, outdir, xlabel, xlim, rrange, lumi, com, logy, shape_norm):
    hists = [h_central, h_up, h_down]

    if shape_norm:
        h_up_ratio = scale_hist_to_ref_integral(h_up, h_central)
        h_down_ratio = scale_hist_to_ref_integral(h_down, h_central)
    else:
        h_up_ratio = h_up
        h_down_ratio = h_down

    hists_ratio = [h_central, h_up_ratio, h_down_ratio]

    labels = [
        "central",
        r"$\alpha_s$ up",
        r"$\alpha_s$ down",
    ]

    colors = MC_COLORS[1:4]

    fig = plot_tools.makePlotWithRatioToRef(
        hists,
        labels,
        colors=colors,
        hists_ratio=hists_ratio,
        linestyles=["solid", "solid", "solid"],
        xlabel=xlabel if xlabel else histname,
        ylabel="Events/bin",
        rlabel=["Var./central"],
        rrange=[rrange],
        xlim=xlim,
        logy=logy,
        baseline=True,
        dataIdx=None,
        ratio_to_data=False,
        yerr=False,
        ratio_legend=False,
        nlegcols=1,
        legtext_size=18,
        cms_label=None,
        lumi=lumi,
        width_scale=1.0,
        linewidth=2,
    )

    hep.cms.label(
        ax=fig.axes[0],
        label="Preliminary",
        data=True,
        lumi=lumi,
        com=com,
        loc=0,
    )

    fig.axes[0].set_xlabel("")
    fig.axes[0].xaxis.label.set_visible(False)

    ratio_ax = fig.axes[-1]

    lower_handles = [
        Line2D([0], [0], color=colors[1], linestyle="solid", linewidth=2),
        Line2D([0], [0], color=colors[2], linestyle="solid", linewidth=2),
    ]

    lower_labels = [
        r"$\alpha_s$ up / central",
        r"$\alpha_s$ down / central",
    ]

    plot_tools.addLegend(
        ratio_ax,
        ncols=1,
        loc="upper right",
        text_size=14,
        extra_handles=lower_handles,
        extra_labels=lower_labels,
    )

    fig.tight_layout()

    basename = f"{histname}"
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)


def make_pdf_plot(h_central, h_variations, histname, outdir, xlabel, xlim, rrange, lumi, com, logy, shape_norm, pdf_lines=False):
    ratios = np.array([
        make_ratio_values(h_var, h_central, shape_norm=shape_norm)
        for h_var in h_variations
    ])

    ratio_min = np.nanmin(ratios, axis=0)
    ratio_max = np.nanmax(ratios, axis=0)

    edges = np.asarray(h_central.axes[0].edges, dtype=float)

    fig = plot_tools.makePlotWithRatioToRef(
        [h_central],
        ["central"],
        colors=[MC_COLORS[0]],
        hists_ratio=[h_central],
        linestyles=["solid"],
        xlabel=xlabel if xlabel else histname,
        ylabel="Events/bin",
        rlabel=["PDF/central"],
        rrange=[rrange],
        xlim=xlim,
        logy=logy,
        baseline=True,
        dataIdx=None,
        ratio_to_data=False,
        yerr=False,
        ratio_legend=False,
        nlegcols=1,
        legtext_size=18,
        cms_label=None,
        lumi=lumi,
        width_scale=1.0,
        linewidth=2,
    )

    hep.cms.label(
        ax=fig.axes[0],
        label="Preliminary",
        data=True,
        lumi=lumi,
        com=com,
        loc=0,
    )

    fig.axes[0].set_xlabel("")
    fig.axes[0].xaxis.label.set_visible(False)

    ratio_ax = fig.axes[-1]

    ratio_ax.axhline(1.0, color="black", linewidth=1.0, zorder=3)

    if pdf_lines:
        for ratio in ratios:
            ratio_ax.step(
                edges,
                np.r_[ratio, ratio[-1]],
                where="post",
                color="gray",
                alpha=0.25,
                linewidth=0.6,
                zorder=1,
            )

    ratio_ax.fill_between(
        edges,
        np.r_[ratio_min, ratio_min[-1]],
        np.r_[ratio_max, ratio_max[-1]],
        step="post",
        color=MC_COLORS[1],
        alpha=0.45,
    )

    ratio_ax.set_ylim(*rrange)

    lower_handles = []
    lower_labels = []

    if pdf_lines:
        lower_handles.append(
            Line2D([0], [0], color="gray", alpha=0.6, linewidth=1.0)
        )
        lower_labels.append("PDF members")

    lower_handles.append(
        Patch(facecolor=MC_COLORS[1], alpha=0.35 if pdf_lines else 0.45)
    )
    lower_labels.append("PDF envelope")

    plot_tools.addLegend(
        ratio_ax,
        ncols=1,
        loc="upper right",
        text_size=14,
        extra_handles=lower_handles,
        extra_labels=lower_labels,
    )

    fig.tight_layout()

    basename = f"{histname}"
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)


def make_variation_plot(infile, outdir, histname, samples, variation, xlabel, xlim, rrange, lumi, com, logy, shape_norm, pdf_lines=False):
    os.makedirs(outdir, exist_ok=True)

    results = load_results(infile)
    print("Available samples:", samples_in_file(results))
    print(f"Using {variation} samples:", samples)

    histname = resolve_variation_histname(histname, variation)
    axis_name = variation_axis_name(variation)

    labels, hists_by_label = sum_variation_hists(results, samples, histname, axis_name)

    central_label = labels[0]
    h_central = hists_by_label[central_label]

    if len(h_central.axes) != 1:
        raise RuntimeError(
            f"{histname} has {len(h_central.axes)} axes after variation slicing. "
            "Use an unrolled plotting script for 2D histograms."
        )

    if variation == "alphaS":
        required = ["central", "alphaSUp", "alphaSDown"]
        missing = [label for label in required if label not in hists_by_label]

        if missing:
            raise RuntimeError(f"Missing alphaS labels {missing}. Available labels are {labels}")

        make_alphaS_plot(
            hists_by_label["central"],
            hists_by_label["alphaSUp"],
            hists_by_label["alphaSDown"],
            histname,
            outdir,
            xlabel,
            xlim,
            rrange,
            lumi,
            com,
            logy,
            shape_norm,
        )

    elif variation == "pdf":
        h_variations = [hists_by_label[label] for label in labels[1:]]

        if len(h_variations) == 0:
            raise RuntimeError(f"No PDF variations found in {histname}. Labels are {labels}")

        make_pdf_plot(
            h_central,
            h_variations,
            histname,
            outdir,
            xlabel,
            xlim,
            rrange,
            lumi,
            com,
            logy,
            shape_norm,
            pdf_lines=pdf_lines,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="HDF5 output from mw_5TeV.py")
    parser.add_argument("--hist", required=True, help="Histogram name, e.g. w_pt, w_mt, met_pt")
    parser.add_argument("-o", "--outdir", default="plots_mw_5TeV")
    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--rrange", nargs=2, type=float, default=[0.95, 1.05])
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--lumi", type=float, default=0.300)
    parser.add_argument("--com", default="5.02")
    parser.add_argument(
        "--mode",
        choices=["dataMC", "alphaS", "pdf"],
        default="dataMC",
        help="dataMC for nominal plots, alphaS/pdf for variation templates",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="MC samples to use for variation plotting",
    )
    parser.add_argument(
        "--rawRatio",
        action="store_true",
        help="Use raw variation/central ratios instead of shape-normalized ratios",
    )
    parser.add_argument(
        "--pdfLines",
        action="store_true",
        help="Draw individual PDF-member ratio lines behind the PDF envelope",
    )

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

    elif args.mode in ["alphaS", "pdf"]:
        samples_for_variation = args.samples

        if samples_for_variation is None or len(samples_for_variation) == 0:
            samples_for_variation = [
                "Wplusmunu_2017G",
                "Wminusmunu_2017G",
                "Wmunu",
            ]

        make_variation_plot(
            args.infile,
            args.outdir,
            args.hist,
            samples_for_variation,
            args.mode,
            args.xlabel,
            args.xlim,
            args.rrange,
            args.lumi,
            args.com,
            args.logy,
            shape_norm=not args.rawRatio,
            pdf_lines=args.pdfLines,
        )


if __name__ == "__main__":
    main()
import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wremnants.utilities.io_tools import input_tools
from wums import boostHistHelpers as hh
from wums import plot_tools


MC_GROUPS = {
    r"$Z/\gamma^*\rightarrow\mu\mu$": ["Zmumu_2017G", "Zmumu"],
    r"$Z/\gamma^*\rightarrow\tau\tau$": ["Ztautau_2017G", "Ztautau"],
    r"$W\rightarrow\tau\nu$": ["Wplustaunu_2017G","Wminustaunu_2017G", "Wtaunu"],
    r"$W\rightarrow\mu\nu$": ["Wplusmunu_2017G", "Wminusmunu_2017G", "Wmunu"],
}

MC_COLORS = [
    "#7DB7E8",  #  blue
    "#F4A259",  # orange
    "#7BC67B",  # green
    "#C44E52",  # red
]

def load_results(infile):
    obj = input_tools.read_infile(infile)
    return obj[0] if isinstance(obj, tuple) else obj

def samples_in_file(results):
    return [k for k in results.keys() if k != "meta_info"]


def read_hist(results, sample, histname):
    return results[sample]["output"][histname].get()


def hist_yield(h):
    y = h.sum()
    return y.value if hasattr(y, "value") else y


def sum_hists(results, samples, histname):
    hsum = None

    for sample in samples:
        if sample not in results:
            continue

        try:
            h = read_hist(results, sample, histname)
        except Exception:
            continue

        print(f"{sample:25s} {histname:20s} yield = {hist_yield(h)}")
        hsum = h.copy() if hsum is None else hsum + h

    return hsum


def get_data_charge_hists(results):
    if "Data" in results:
        data_sample = "Data"
    else:
        data_samples = [s for s in samples_in_file(results) if s.startswith("SingleMuon")]
        if len(data_samples) != 1:
            raise RuntimeError(f"Expected one data sample, found {data_samples}")
        data_sample = data_samples[0]

    return (
        read_hist(results, data_sample, "wpt_mueta_minus"),
        read_hist(results, data_sample, "wpt_mueta_plus"),
        "Data",
    )

def get_mc_charge_stacks(results):
    minus_hists = []
    plus_hists = []
    labels = []

    for label, samples in MC_GROUPS.items():
        h_minus = sum_hists(results, samples, "wpt_mueta_minus")
        h_plus = sum_hists(results, samples, "wpt_mueta_plus")

        if h_minus is None or h_plus is None:
            continue

        minus_hists.append(h_minus)
        plus_hists.append(h_plus)
        labels.append(label)

    if not minus_hists:
        raise RuntimeError("No MC charge-split histograms found.")

    return minus_hists, plus_hists, labels


def unroll(h2d):
    return hh.unrolledHist(h2d, obs=["w_pt", "mu_eta"])


def draw_mc_panel(ax, h2d_list, labels, colors, charge_label):
    hists = [unroll(h) for h in h2d_list]

    hep.histplot(
        hists,
        ax=ax,
        stack=True,
        histtype="fill",
        color=colors[:len(hists)],
        alpha=0.65,
        linewidth=0,
        label=labels,
        flow="none",
    )

    hep.histplot(
        hists,
        ax=ax,
        stack=True,
        histtype="step",
        color="black",
        linewidth=1.0,
        flow="none",
    )

    ax.text(0.45, 0.86, charge_label, transform=ax.transAxes, fontsize=16)
    ax.set_ylabel("Events/bin", fontsize=13)
    ax.set_xlabel("")
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    plot_tools.addLegend(ax, ncols=1, loc="upper right", text_size="small")


def draw_data_panel(ax, h2d, charge_label, source_label):
    h = unroll(h2d)
    hep.histplot(
        h,
        ax=ax,
        histtype="errorbar",
        color="black",
        label=source_label,
        flow="none",
    )

    ax.text(0.035, 0.86, charge_label, transform=ax.transAxes, fontsize=12)
    ax.set_ylabel("Events/bin", fontsize=13)
    ax.set_xlabel("")   # <- important
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    plot_tools.addLegend(ax, ncols=1, loc="upper right", text_size="small")


def make_plot(infile, outdir, mode, lumi, com, xlim, logy):
    os.makedirs(outdir, exist_ok=True)

    results = load_results(infile)
    print("Available samples/groups:", samples_in_file(results))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"hspace": 0.10},
    )

    if mode == "mc":
        h_minus_list, h_plus_list, labels = get_mc_charge_stacks(results)

        draw_mc_panel(
            axes[0],
            h_minus_list,
            labels,
            MC_COLORS,
            r"Charge = -1",
        )

        draw_mc_panel(
            axes[1],
            h_plus_list,
            labels,
            MC_COLORS,
            r"Charge = +1",
        )

        basename = "unrolled_wpt_mueta_chargeSplit"

    else:
        h_minus_2d, h_plus_2d, source_label = get_data_charge_hists(results)

        draw_data_panel(
            axes[0],
            h_minus_2d,
            r"charge = -1",
            source_label,
        )

        draw_data_panel(
            axes[1],
            h_plus_2d,
            r"charge = +1",
            source_label,
        )

        basename = "unrolled_wpt_mueta_chargeSplit_data"

    # Hide top-panel x tick labels without breaking shared-axis formatting
    axes[0].tick_params(axis="x", labelbottom=False)

    # Make sure only the bottom panel carries the x label
    axes[0].set_xlabel("")
    axes[1].set_xlabel(r"$(p_T^W, \eta^{\mu})$ bin", fontsize=14)

    # Bottom-axis range and ticks
    axes[1].set_xlim(*xlim)
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(10, integer=True))
    axes[1].tick_params(axis="x", labelbottom=True)

    if logy:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")

    hep.cms.label(
        ax=axes[0],
        label="Preliminary",
        data=True,
        lumi=lumi,
        com=com,
        loc=0,
    )

    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.12, hspace=0.12)
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="HDF5 output from mw_5TeV.py")
    parser.add_argument("-o", "--outdir", default="plots_mw_5TeV")
    parser.add_argument("--mode", choices=["data", "mc"], default="mc")
    parser.add_argument("--lumi", type=float, default=0.300, help="Luminosity in fb^-1")
    parser.add_argument("--com", default="5.02", help="Center-of-mass energy in TeV")
    parser.add_argument("--logy", action="store_true")
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=[0, 120],
        help="x-axis range in unrolled bin index",
    )

    args = parser.parse_args()

    make_plot(
        args.infile,
        args.outdir,
        args.mode,
        args.lumi,
        args.com,
        args.xlim,
        args.logy,
    )


if __name__ == "__main__":
    main()
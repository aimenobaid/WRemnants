import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep

from wremnants.utilities.io_tools import input_tools
from wums import plot_tools


MC_GROUPS = {
    r"$Z/\gamma^*\rightarrow\mu\mu$": ["Zmumu_2017G"],
    r"$Z/\gamma^*\rightarrow\tau\tau$": ["Ztautau_2017G"],
    r"$W\rightarrow\tau\nu$": ["Wminustaunu_2017G"],
    r"$W\rightarrow\mu\nu$": ["Wplusmunu_2017G", "Wminusmunu_2017G"],
}

def read_hist(infile, sample, histname):
    return input_tools.read_and_scale(infile, sample, histname)


def samples_in_file(infile):
    return [k for k in input_tools.read_keys(infile) if k != "meta_info"]

def sum_hists(infile, samples, histname):
    hsum = None

    for sample in samples:
        h = read_hist(infile, sample, histname)
        y = h.sum().value if hasattr(h.sum(), "value") else h.sum()
        print(f"{sample:25s} {histname:15s} yield = {y}")

        hsum = h.copy() if hsum is None else hsum + h

    return hsum

# def get_data_hist(infile, histname):
#     data_samples = [s for s in samples_in_file(infile) if s.startswith("SingleMuon")]
#     if len(data_samples) != 1:
#         raise RuntimeError(f"Expected one SingleMuon sample, found {data_samples}")

#     h = read_hist(infile, data_samples[0], histname)
#     y = h.sum().value if hasattr(h.sum(), "value") else h.sum()
#     print(f"{data_samples[0]:25s} {histname:15s} yield = {y}")
#     return h

def get_mc_stack(infile, histname):
    hists = []
    labels = []

    for label, samples in MC_GROUPS.items():
        h = sum_hists(infile, samples, histname)
        if h is None:
            continue

        hists.append(h)
        labels.append(label)

    if not hists:
        raise RuntimeError(f"No MC histograms found for {histname}")

    return hists, labels

def make_plot(infile, outdir, histname, xlabel, xlim, lumi, com, logy):
    os.makedirs(outdir, exist_ok=True)

    mc_hists, mc_labels = get_mc_stack(infile, histname)
    # data_hist = get_data_hist(infile, histname)

    fig, ax = plt.subplots(figsize=(10.5, 8.0))

    hep.histplot(
        mc_hists,
        ax=ax,
        stack=True,
        histtype="fill",
        label=mc_labels,
        flow="none",
    )

    # hep.histplot(
    #     data_hist,
    #     ax=ax,
    #     histtype="errorbar",
    #     color="black",
    #     label="Data",
    #     flow="none",
    # )

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

    basename = f"{histname}_stacked_MC"
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

    args = parser.parse_args()

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

if __name__ == "__main__":
    main()
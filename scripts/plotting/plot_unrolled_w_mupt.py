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


def read_hist(infile, sample, histname):
    return input_tools.read_and_scale(infile, sample, histname)

def find_data_sample(infile):
    samples = [k for k in input_tools.read_keys(infile) if k != "meta_info"]
    data_samples = [s for s in samples if s.startswith("SingleMuon")]

    if len(data_samples) != 1:
        raise RuntimeError(f"Expected one SingleMuon data sample, found {data_samples}")

    return data_samples[0]

def find_mc_samples(infile):
    samples = [k for k in input_tools.read_keys(infile) if k != "meta_info"]
    return [s for s in samples if not s.startswith("SingleMuon")]


def sum_hists(infile, samples, histname):
    hsum = None

    for sample in samples:
        try:
            h = read_hist(infile, sample, histname)
        except Exception:
            continue

        hsum = h.copy() if hsum is None else hsum + h

    if hsum is None:
        raise RuntimeError(f"Could not read {histname} from samples {samples}")

    return hsum


def get_charge_hists(infile, mode):
    if mode == "data":
        data_sample = find_data_sample(infile)
        return (
            read_hist(infile, data_sample, "mupt_absEta_minus"),
            read_hist(infile, data_sample, "mupt_absEta_plus"),
            "Data",
        )

    # For the first validation plot, use signal MC only.
    # Backgrounds like Zmumu, Ztautau, and Wtaunu should be added later
    # as separate stacked components.
    signal_samples = [
        s for s in find_mc_samples(infile)
        if "munu" in s.lower() and "tau" not in s.lower()
    ]

    return (
        sum_hists(infile, signal_samples, "mupt_absEta_minus"),
        sum_hists(infile, signal_samples, "mupt_absEta_plus"),
        r"$W\rightarrow\mu\nu$ signal MC",
    )


def unroll(h2d):
    return hh.unrolledHist(h2d, obs=["mu_pt", "abs_mu_eta"])


def last_nonzero_bin(*hists):
    last = 0

    for h in hists:
        values = h.values(flow=False)
        nonzero = np.nonzero(values > 0)[0]

        if len(nonzero):
            last = max(last, int(nonzero[-1]))

    return last


def draw_panel(ax, h2d, charge_label, source_label):
    h = unroll(h2d)

    hep.histplot(
        h,
        ax=ax,
        histtype="fill",
        color="tab:red",
        alpha=0.9,
        linewidth=0,
        flow="none",
        label=source_label,
    )

    hep.histplot(
        h,
        ax=ax,
        histtype="step",
        color="black",
        linewidth=0.8,
        flow="none",
    )

    ax.text(0.035, 0.86, charge_label, transform=ax.transAxes, fontsize=12)

    ax.set_ylabel("Events/bin", fontsize=13)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.minorticks_on()

    plot_tools.addLegend(ax, ncols=1, loc="upper right", text_size="small")

    return h


def make_plot(infile, outdir, mode, lumi, com):
    os.makedirs(outdir, exist_ok=True)

    h_minus_2d, h_plus_2d, source_label = get_charge_hists(infile, mode)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 6.8),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    h_minus = draw_panel(
        axes[0],
        h_minus_2d,
        r"charge = -1",
        source_label,
    )

    h_plus = draw_panel(
        axes[1],
        h_plus_2d,
        r"charge = +1",
        source_label,
    )

    xmax = last_nonzero_bin(h_minus, h_plus) + 5

    axes[0].set_xticklabels([])
    axes[1].set_xlim(0, xmax)
    axes[1].set_xlabel(r"Reco $(p_T^\mu, |\eta^\mu|)$ bin", fontsize=14)
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(8))

    # Force the CMS label not to print "Simulation".
    hep.cms.label(
        ax=axes[0],
        label="Preliminary",
        data=True,
        lumi=lumi,
        com=com,
        loc=0,
    )

    fig.tight_layout()

    basename = f"unrolled_reco_mupt_absEta_chargeSplit_{mode}"
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="HDF5 output from mw_5TeV.py")
    parser.add_argument("-o", "--outdir", default="plots_mw_5TeV_unrolled")
    parser.add_argument("--mode", choices=["data", "mc"], default="mc")
    parser.add_argument("--lumi", type=float, default=0.300, help="Luminosity in fb^-1")
    parser.add_argument("--com", default="5.02", help="Center-of-mass energy in TeV")

    args = parser.parse_args()

    make_plot(args.infile, args.outdir, args.mode, args.lumi, args.com)


if __name__ == "__main__":
    main()
import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wremnants.utilities.io_tools import input_tools
from wums import plot_tools


# Only use W -> mu nu.
# Prefer explicit W+ and W- samples. Fall back to an aggregate Wmunu sample only if needed.
WPLUS_FILTERS = ["Wplusmunu_2017G", "Wplusmunu"]
WMINUS_FILTERS = ["Wminusmunu_2017G", "Wminusmunu"]
WAGG_FILTERS = ["Wmunu"]


def load_results(infile):
    obj = input_tools.read_infile(infile)
    return obj[0] if isinstance(obj, tuple) else obj


def samples_in_file(results):
    return [k for k in results.keys() if k != "meta_info"]


def read_hist(results, sample, histname):
    return results[sample]["output"][histname].get()


def sample_matches(sample, filters):
    return any(sample == filt or sample.startswith(filt) for filt in filters)


def find_samples(results, filters):
    samples = samples_in_file(results)
    return [s for s in samples if sample_matches(s, filters)]


def sum_hists(results, samples, histname):
    hsum = None

    for sample in samples:
        if histname not in results[sample]["output"]:
            print(f"Skipping {sample}: missing histogram {histname}")
            continue

        h = read_hist(results, sample, histname)
        y = h.sum().value if hasattr(h.sum(), "value") else h.sum()
        print(f"{sample:25s} {histname:25s} yield = {y}")

        hsum = h.copy() if hsum is None else hsum + h

    return hsum


def get_wmunu_hist(results, histname, split_charges=False):
    wplus_samples = find_samples(results, WPLUS_FILTERS)
    wminus_samples = find_samples(results, WMINUS_FILTERS)

    if wplus_samples or wminus_samples:
        print("Using explicit W+ and W- samples.")
        print("W+ samples:", wplus_samples)
        print("W- samples:", wminus_samples)

        h_plus = sum_hists(results, wplus_samples, histname) if wplus_samples else None
        h_minus = sum_hists(results, wminus_samples, histname) if wminus_samples else None

        if split_charges:
            out = []
            if h_plus is not None:
                out.append((r"$W^+\rightarrow\mu\nu$", h_plus))
            if h_minus is not None:
                out.append((r"$W^-\rightarrow\mu\nu$", h_minus))
            return out

        h_total = None
        for h in [h_plus, h_minus]:
            if h is None:
                continue
            h_total = h.copy() if h_total is None else h_total + h

        if h_total is None:
            raise RuntimeError(f"No Wmunu histograms found for {histname}")

        return [(r"$W^\pm\rightarrow\mu\nu$", h_total)]

    # Fallback only if explicit plus/minus are not present.
    wagg_samples = find_samples(results, WAGG_FILTERS)

    if not wagg_samples:
        raise RuntimeError(
            f"No Wplusmunu/Wminusmunu/Wmunu samples found. Available samples: {samples_in_file(results)}"
        )

    print("Using aggregate Wmunu samples:", wagg_samples)
    h_total = sum_hists(results, wagg_samples, histname)

    if h_total is None:
        raise RuntimeError(f"No Wmunu histograms found for {histname}")

    return [(r"$W\rightarrow\mu\nu$", h_total)]


def axis_bin_label(axis, axis_name, ibin):
    lo = float(axis.edges[ibin])
    hi = float(axis.edges[ibin + 1])

    if axis_name == "abs_dxy":
        return rf"$|d_{{xy}}|\in[{lo:g},{hi:g})$ cm"

    if axis_name == "relIso":
        return rf"RelIso $\in[{lo:g},{hi:g})$"

    return rf"{axis_name} $\in[{lo:g},{hi:g})$"


def slice_to_1d(h, x_axis, slice_axis, ibin):
    h_slice = h[{slice_axis: ibin}]

    if len(h_slice.axes) > 1:
        h_slice = h_slice.project(x_axis)

    axis_names = list(h_slice.axes.name)

    if len(axis_names) != 1 or axis_names[0] != x_axis:
        raise RuntimeError(
            f"After slicing {slice_axis}:{ibin}, expected one axis {x_axis}, got {axis_names}"
        )

    return h_slice


def hist_integral(h):
    s = h.sum()
    return s.value if hasattr(s, "value") else s


def normalize_hist(h):
    integral = hist_integral(h)

    if integral == 0:
        return h.copy()

    return h * (1.0 / integral)

def get_step_values(h, density=False, normalize=False):
    edges = np.asarray(h.axes[0].edges, dtype=float)
    values = np.asarray(h.values(flow=False), dtype=float)
    widths = np.diff(edges)

    total = np.sum(values)

    if density:
        if total > 0:
            values = values / (total * widths)
    elif normalize:
        if total > 0:
            values = values / total

    return edges, values

def make_line_slices_plot(
    infile,
    outdir,
    histname,
    x_axis,
    slice_axis,
    xlabel,
    ylabel,
    xlim,
    ylim,
    lumi,
    com,
    logy,
    normalize,
    density,
    split_charges,
):
    os.makedirs(outdir, exist_ok=True)

    results = load_results(infile)
    print("Available samples:", samples_in_file(results))

    w_hists = get_wmunu_hist(results, histname, split_charges=split_charges)

    # Use the first W histogram to validate axes.
    ref_hist = w_hists[0][1]
    axis_names = list(ref_hist.axes.name)

    print("Histogram axes:", axis_names)

    if len(axis_names) != 2:
        raise RuntimeError(
            f"Histogram {histname} has axes {axis_names}. "
            "This line-slice plot expects a 2D histogram, e.g. ['abs_dxy', 'relIso']."
        )

    if x_axis not in axis_names:
        raise RuntimeError(f"xAxis {x_axis} not found. Available axes are {axis_names}")

    if slice_axis not in axis_names:
        raise RuntimeError(f"sliceAxis {slice_axis} not found. Available axes are {axis_names}")

    if x_axis == slice_axis:
        raise RuntimeError("xAxis and sliceAxis must be different.")

    slice_axis_obj = ref_hist.axes[slice_axis]
    n_slice_bins = len(slice_axis_obj)

    fig, ax = plt.subplots(figsize=(11.5, 8.5))

    linestyles = ["solid", "dashed", "dashdot", "dotted"]

    for iw, (w_label, h2d) in enumerate(w_hists):
        for ibin in range(n_slice_bins):
            h1d = slice_to_1d(h2d, x_axis, slice_axis, ibin)

            label = axis_bin_label(slice_axis_obj, slice_axis, ibin)

            if split_charges:
                label = f"{w_label}, {label}"

            linestyle = linestyles[iw % len(linestyles)]

            edges, values = get_step_values(
                h1d,
                density=density,
                normalize=normalize,
            )

            ax.step(
                edges,
                np.r_[values, values[-1]],
                where="post",
                linewidth=1.8,
                linestyle=linestyle,
                label=label,
            )

    ax.set_xlabel(xlabel if xlabel else x_axis, fontsize=14)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14)
    elif density:
        ax.set_ylabel("Normalized density", fontsize=14)
    elif normalize:
        ax.set_ylabel("Normalized events/bin", fontsize=14)
    else:
        ax.set_ylabel("Events/bin", fontsize=14)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if logy:
        ax.set_yscale("log")

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.minorticks_on()

    plot_tools.addLegend(
        ax,
        ncols=2,
        loc="upper right",
        text_size=10,
    )

    hep.cms.label(
        ax=ax,
        label="Preliminary",
        data=False,
        lumi=lumi,
        com=com,
        loc=0,
    )

    fig.tight_layout()

    if density:
        norm_tag = "density"
    elif normalize:
        norm_tag = "shapeNorm"
    else:
        norm_tag = "yield"
    split_tag = "splitW" if split_charges else "sumW"
    basename = f"{histname}_{x_axis}_lines_by_{slice_axis}_{split_tag}_{norm_tag}"

    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="HDF5 output from mw_5TeV.py")
    parser.add_argument("--hist", default="dxy_vs_relIso_shape")
    parser.add_argument("-o", "--outdir", default="plots_mw_5TeV/dxy_relIso_lines")

    parser.add_argument(
        "--mode",
        choices=["relIsoByDxy", "dxyByRelIso", "both"],
        default="both",
        help=(
            "relIsoByDxy: x-axis RelIso, one line per abs_dxy bin. "
            "dxyByRelIso: x-axis abs_dxy, one line per RelIso bin."
        ),
    )

    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--normalize", action="store_true", help="Normalize each line to unit area.")
    parser.add_argument("--density",action="store_true",
        help="Normalize each line to unit area and divide by bin width.")
    parser.add_argument("--splitWCharges",action="store_true",help="Draw W+ and W- separately. Default is to sum W+ and W-.")
    parser.add_argument("--lumi", type=float, default=0.300)
    parser.add_argument("--com", default="5.02")

    args = parser.parse_args()

    if args.mode in ["relIsoByDxy", "both"]:
        outdir = args.outdir
        if args.mode == "both":
            outdir = os.path.join(args.outdir, "relIso_by_dxy")

        make_line_slices_plot(
            args.infile,
            outdir,
            args.hist,
            x_axis="relIso",
            slice_axis="abs_dxy",
            xlabel=args.xlabel if args.mode != "both" else "RelIso",
            ylabel=args.ylabel,
            xlim=args.xlim if args.mode != "both" else [0.0, 1.0],
            ylim=args.ylim,
            lumi=args.lumi,
            com=args.com,
            logy=args.logy,
            normalize=args.normalize,
            density=args.density,
            split_charges=args.splitWCharges,
        )

    if args.mode in ["dxyByRelIso", "both"]:
        outdir = args.outdir
        if args.mode == "both":
            outdir = os.path.join(args.outdir, "dxy_by_relIso")

        make_line_slices_plot(
            args.infile,
            outdir,
            args.hist,
            x_axis="abs_dxy",
            slice_axis="relIso",
            xlabel=args.xlabel if args.mode != "both" else r"$|d_{xy}|$ [cm]",
            ylabel=args.ylabel,
            xlim=args.xlim if args.mode != "both" else [0.0, 0.05],
            ylim=args.ylim,
            lumi=args.lumi,
            com=args.com,
            logy=args.logy,
            normalize=args.normalize,
            density=args.density,
            split_charges=args.splitWCharges,
        )

if __name__ == "__main__":
    main()
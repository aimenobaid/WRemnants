import argparse
import h5py
import os
import sys
import hist
import numpy as np


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


wremnants_base = os.environ.get("WREM_BASE", None)
if wremnants_base is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wremnants_base = os.path.abspath(os.path.join(script_dir, "../.."))
    if os.path.exists(os.path.join(wremnants_base, "wums")):
        sys.path.insert(0, wremnants_base)

from wremnants.utilities.io_tools import base_io
from rabbit import tensorwriter

# -----------------------------------
# Arguments
# -----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("infile", help="Input WRemnants HDF5 file")
parser.add_argument("-o", "--output", default="./", help="Output directory")
parser.add_argument("--outname", default="my_tensor", help="Output tensor file name")
parser.add_argument("--histName", default="ptll", help="Nominal histogram name")
parser.add_argument(
    "--analysis",
    choices=["z", "w"],
    default="z",
    help="Analysis mode. Only controls default process filters.",
)
parser.add_argument(
    "--procFilters",
    nargs="*",
    default=None,
    help="Processes to include. Defaults depend on --analysis.",
)
parser.add_argument(
    "--signalFilters",
    nargs="*",
    default=None,
    help="Processes receiving PDF/alphaS systematics. Defaults depend on --analysis.",
)
parser.add_argument(
    "--asimovMode",
    choices=["first", "sum"],
    default="first",
    help="first = old behavior, use first MC process as Asimov data; sum = sum selected MC processes.",
)
parser.add_argument("--sparse", default=False, action="store_true", help="Make sparse tensor")
parser.add_argument(
    "--systematicType",
    choices=["log_normal", "normal"],
    default="log_normal",
    help="Probability density for systematic variations",
)
parser.add_argument(
    "--alphaSUpName",
    default="pdfCT18ZNNLO_as_0120",
    help="Name of the alphaS-up variation on the vars axis",
)
parser.add_argument(
    "--alphaSDownName",
    default="pdfCT18ZNNLO_as_0116",
    help="Name of the alphaS-down variation on the vars axis",
)
parser.add_argument(
    "--wChargeAxis",
    action="store_true",
    help=(
        "For W only: build a single tensor with a reco charge axis by combining "
        "<histName>_minus and <histName>_plus into one histogram."
    ),
)

# ABCD / data-driven nonprompt options
parser.add_argument(
    "--useDataObs",
    action="store_true",
    help="Use SingleMuon data as observed data instead of Asimov MC.",
)
parser.add_argument(
    "--dataProcFilter",
    default="SingleMuon",
    help="Substring used to find the data process.",
)
parser.add_argument(
    "--addNonprompt",
    action="store_true",
    help="Add a synthetic nonprompt process built from data minus selected MC.",
)
parser.add_argument(
    "--nonpromptName",
    default="nonprompt",
    help="Name of the synthetic nonprompt process.",
)
parser.add_argument(
    "--nonpromptFloor",
    type=float,
    default=1e-6,
    help="Minimum bin content for the nonprompt template.",
)

args = parser.parse_args()

# ---------------------------
# Defaults
# ---------------------------
if args.procFilters is None:
    if args.analysis == "z":
        args.procFilters = ["Zmumu"]
    else:
        args.procFilters = ["Wplusmunu", "Wminusmunu", "Wmunu"]

if args.signalFilters is None:
    if args.analysis == "z":
        args.signalFilters = ["Zmumu", "Ztautau"]
    else:
        args.signalFilters = ["Wplusmunu", "Wminusmunu", "Wmunu"]

# -------------------
# helpers
# -------------------
def get_hist(results, proc, hist_name):
    h_proxy = results[proc]["output"][hist_name]
    return h_proxy.get() if hasattr(h_proxy, "get") else h_proxy


def has_hist(results, proc, hist_name):
    return (
        proc in results
        and "output" in results[proc]
        and hist_name in results[proc]["output"]
    )


def add_hist(hsum, h):
    return h.copy() if hsum is None else hsum + h


def find_one_process(processes, filt):
    matches = [p for p in processes if filt in p]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one process matching '{filt}', found {matches}"
        )

    return matches[0]


def clip_hist_to_floor(h, floor):
    h = h.copy()
    view = h.view(flow=False)

    if hasattr(view, "value"):
        mask = view.value <= floor
        view.value[mask] = floor

        if hasattr(view, "variance") and view.variance is not None:
            view.variance[mask] = floor
    else:
        view[...] = np.where(view > floor, view, floor)

    return h


def make_nonprompt_template(h_data, prompt_hists, floor):
    """
    Build a seed nonprompt template from data minus selected MC.

    This intentionally edits the weighted-histogram values directly, because
    boost-histogram weighted storage supports addition but may not support
    direct histogram subtraction through h1 - h2 / h1 -= h2.
    """
    h_nonprompt = h_data.copy()
    view_np = h_nonprompt.view(flow=False)

    if not hasattr(view_np, "value"):
        raise RuntimeError(
            "Expected weighted histogram storage with .value/.variance for nonprompt template."
        )

    for proc, h_prompt in prompt_hists.items():
        print_flush(f"Subtracting MC from nonprompt seed: {proc}")

        if h_prompt.axes.name != h_nonprompt.axes.name:
            raise RuntimeError(
                f"Axis mismatch while building nonprompt for {proc}: "
                f"data axes={h_nonprompt.axes.name}, prompt axes={h_prompt.axes.name}"
            )

        view_prompt = h_prompt.view(flow=False)

        if not hasattr(view_prompt, "value"):
            raise RuntimeError(
                f"Prompt histogram for {proc} does not have weighted storage."
            )

        view_np.value[...] = view_np.value - view_prompt.value

        if view_np.variance is not None and view_prompt.variance is not None:
            # Variance of a difference adds: Var(data - MC) = Var(data) + Var(MC)
            view_np.variance[...] = view_np.variance + view_prompt.variance

    mask = view_np.value <= floor
    view_np.value[mask] = floor

    if view_np.variance is not None:
        view_np.variance[mask] = floor

    total = h_nonprompt.sum().value if hasattr(h_nonprompt.sum(), "value") else h_nonprompt.sum()
    print_flush(f"Built nonprompt seed template with total yield = {total}")

    return h_nonprompt


def project_to_nominal_axes(h_var, h_nom):
    return h_var.project(*h_nom.axes.name)


def find_corr_hist_name(results, proc, hist_name, tag):
    output = results[proc]["output"]
    matches = [
        name
        for name in output
        if name.startswith(f"{hist_name}_")
        and tag in name
        and name.endswith("_Corr")
    ]

    if matches:
        if len(matches) > 1:
            print_flush(
                f"Warning: multiple {tag} histograms matched histName={hist_name} "
                f"for {proc}: {matches}. Using {matches[0]}"
            )
        return matches[0]

    # Fallback keeps older files usable if their naming is less strict.
    fallback_matches = [
        name
        for name in output
        if tag in name and name.endswith("_Corr")
    ]

    if fallback_matches:
        print_flush(
            f"Warning: no {tag} histogram matched histName={hist_name} for {proc}. "
            f"Falling back to {fallback_matches[0]}"
        )
        return fallback_matches[0]

    return None


def load_corr_hists(results, signal_procs, hist_name, tag):
    hists = {}
    names = {}

    for proc in signal_procs:
        corr_name = find_corr_hist_name(results, proc, hist_name, tag)
        if corr_name is None:
            print_flush(
                f"Warning: no {tag} _Corr histogram found for {proc} "
                f"with histName={hist_name}"
            )
            continue

        hists[proc] = get_hist(results, proc, corr_name)
        names[proc] = corr_name

        print_flush(f"Found {tag} correction histogram for {proc}: {corr_name}")
        print_flush(f"  axes: {[ax.name for ax in hists[proc].axes]}")

    return hists, names


def vars_labels(h):
    return [str(v) for v in h.axes["vars"]]


def is_central_pdf_label(label):
    return label == "central" or label.startswith("pdf0")


def choose_existing_label(labels, candidates):
    for candidate in candidates:
        if candidate in labels:
            return candidate
    return None


def pdf_pairs_from_vars(labels):
    pdf_labels = [
        label
        for label in labels
        if not is_central_pdf_label(label) and "_as_" not in label
    ]

    down_labels = [label for label in pdf_labels if label.endswith("Down")]
    up_labels = [label for label in pdf_labels if label.endswith("Up")]

    if down_labels and up_labels:
        bases = sorted(
            set(
                label[:-4] if label.endswith("Down") else label[:-2]
                for label in down_labels + up_labels
            )
        )

        pairs = []
        for base in bases:
            down = f"{base}Down"
            up = f"{base}Up"

            if down in pdf_labels and up in pdf_labels:
                pairs.append((up, down, base))
            else:
                print_flush(f"Skipping incomplete PDF pair for {base}")

        return pairs

    # Fallback for old label ordering.
    pairs = []
    for up, down in zip(pdf_labels[1::2], pdf_labels[2::2]):
        pairs.append((up, down, f"{up}_{down}"))

    return pairs


def make_charge_axis():
    # Bin 0: charge -1, bin 1: charge +1.
    return hist.axis.Regular(
        2,
        -2.0,
        2.0,
        name="charge",
        underflow=False,
        overflow=False,
    )


def combine_charge_hists(h_minus, h_plus):
    if h_minus.axes.name != h_plus.axes.name:
        raise RuntimeError(
            f"Cannot combine charge hists with different axes: "
            f"minus={h_minus.axes.name}, plus={h_plus.axes.name}"
        )

    axis_charge = make_charge_axis()

    h_charge = hist.Hist(
        *h_minus.axes,
        axis_charge,
        storage=h_minus.storage_type(),
    )

    # charge bin 0 is negative, charge bin 1 is positive
    h_charge.view(flow=False)[..., 0] = h_minus.view(flow=False)
    h_charge.view(flow=False)[..., 1] = h_plus.view(flow=False)

    return h_charge


def load_charge_nominal_hist(results, proc, hist_name):
    minus_name = f"{hist_name}_minus"
    plus_name = f"{hist_name}_plus"

    if not has_hist(results, proc, minus_name):
        print_flush(f"Warning: missing nominal histogram {minus_name} for {proc}")
        return None

    if not has_hist(results, proc, plus_name):
        print_flush(f"Warning: missing nominal histogram {plus_name} for {proc}")
        return None

    h_minus = get_hist(results, proc, minus_name)
    h_plus = get_hist(results, proc, plus_name)

    h_charge = combine_charge_hists(h_minus, h_plus)

    print_flush(
        f"Built charge-axis nominal histogram for {proc}: "
        f"{minus_name} + {plus_name} -> axes {[ax.name for ax in h_charge.axes]}"
    )

    return h_charge


def load_charge_corr_hists(results, signal_procs, hist_name, tag):
    hists = {}
    names = {}

    for proc in signal_procs:
        minus_base = f"{hist_name}_minus"
        plus_base = f"{hist_name}_plus"

        minus_corr_name = find_corr_hist_name(results, proc, minus_base, tag)
        plus_corr_name = find_corr_hist_name(results, proc, plus_base, tag)

        if minus_corr_name is None or plus_corr_name is None:
            print_flush(
                f"Warning: missing charge-split {tag} _Corr histograms for {proc}. "
                f"minus={minus_corr_name}, plus={plus_corr_name}"
            )
            continue

        h_minus = get_hist(results, proc, minus_corr_name)
        h_plus = get_hist(results, proc, plus_corr_name)

        h_charge = combine_charge_hists(h_minus, h_plus)

        hists[proc] = h_charge
        names[proc] = (minus_corr_name, plus_corr_name)

        print_flush(
            f"Built charge-axis {tag} histogram for {proc}: "
            f"{minus_corr_name} + {plus_corr_name} -> axes {[ax.name for ax in h_charge.axes]}"
        )

    return hists, names


# ----------------------------------------------
# load inputs
# ----------------------------------------------
infile_path = os.path.abspath(args.infile)
if not os.path.exists(infile_path):
    raise FileNotFoundError(f"Input file not found: {infile_path}")

h5file = h5py.File(infile_path, "r")
results = base_io.load_results_h5py(h5file)

hist_name = args.histName

all_procs = [p for p in results.keys() if p != "meta_info"]
print_flush(f"All processes found in file: {all_procs}")

if args.procFilters:
    procs_to_use = [
        p for p in all_procs
        if any(filt in p for filt in args.procFilters)
    ]
else:
    procs_to_use = all_procs

mc_procs = procs_to_use

signal_procs = [
    p for p in mc_procs
    if any(filt in p for filt in args.signalFilters)
]

background_procs = [
    p for p in mc_procs
    if p not in signal_procs
]

print_flush(f"Processes after filtering: {procs_to_use}")
print_flush(f"Signal processes: {signal_procs}")
print_flush(f"Background processes: {background_procs}")

# --------------------------
# load nominal histograms
# --------------------------
h_mc_dict = {}

use_w_charge_axis = args.analysis == "w" and args.wChargeAxis

if use_w_charge_axis:
    print_flush(
        f"W charge-axis mode enabled for histName={hist_name}. "
        f"Will combine {hist_name}_minus and {hist_name}_plus into one tensor axis."
    )

for proc in mc_procs:
    if use_w_charge_axis:
        h_charge = load_charge_nominal_hist(results, proc, hist_name)
        if h_charge is not None:
            h_mc_dict[proc] = h_charge
    else:
        if has_hist(results, proc, hist_name):
            h_mc_dict[proc] = get_hist(results, proc, hist_name)
        else:
            print_flush(f"Warning: missing nominal histogram {hist_name} for {proc}")

if not h_mc_dict:
    raise RuntimeError(f"No MC histograms found for histName={hist_name}")

# -----------------------------------------------
# load correction histograms
# -----------------------------------------------
if use_w_charge_axis:
    h_pdfas_corr_dict, pdfas_names = load_charge_corr_hists(
        results,
        signal_procs,
        hist_name,
        "pdfas",
    )

    h_pdfvars_corr_dict, pdfvars_names = load_charge_corr_hists(
        results,
        signal_procs,
        hist_name,
        "pdfvars",
    )
else:
    h_pdfas_corr_dict, pdfas_names = load_corr_hists(
        results,
        signal_procs,
        hist_name,
        "pdfas",
    )

    h_pdfvars_corr_dict, pdfvars_names = load_corr_hists(
        results,
        signal_procs,
        hist_name,
        "pdfvars",
    )

# ------------------------------------------------------------------
# Old Asimov-only behavior kept here for reference.
# Do not uncomment this together with the active block below.
# ------------------------------------------------------------------
# # -------------------
# # Build Asimov data
# # -------------------
# if args.asimovMode == "first":
#     first_mc_proc = list(h_mc_dict.keys())[0]
#     h_data = h_mc_dict[first_mc_proc].copy()
#     print_flush(f"Using MC process '{first_mc_proc}' as expected data")
# else:
#     h_data = None
#     for proc, h in h_mc_dict.items():
#         h_data = add_hist(h_data, h)
#     print_flush(f"Using sum of {len(h_mc_dict)} MC processes as expected data")
#
# # Nominal histograms should not usually have vars, but keep old behavior.
# if hasattr(h_data, "axes") and "vars" in h_data.axes.name:
#     h_data_base = h_data[{"vars": 0}]
#     h_mc_base = {proc: h[{"vars": 0}] for proc, h in h_mc_dict.items()}
# else:
#     h_data_base = h_data
#     h_mc_base = h_mc_dict

# -------------------
# Build observed data / Asimov data
# -------------------
if args.useDataObs or args.addNonprompt:
    data_proc = find_one_process(all_procs, args.dataProcFilter)

    if not has_hist(results, data_proc, hist_name):
        raise RuntimeError(
            f"Data process {data_proc} does not have histogram {hist_name}"
        )

    h_data = get_hist(results, data_proc, hist_name)
    print_flush(f"Using data process '{data_proc}' as observed data")

else:
    if args.asimovMode == "first":
        first_mc_proc = list(h_mc_dict.keys())[0]
        h_data = h_mc_dict[first_mc_proc].copy()
        print_flush(f"Using MC process '{first_mc_proc}' as expected data")
    else:
        h_data = None
        for proc, h in h_mc_dict.items():
            h_data = add_hist(h_data, h)
        print_flush(f"Using sum of {len(h_mc_dict)} MC processes as expected data")

# Nominal histograms should not usually have vars, but keep old behavior.
if hasattr(h_data, "axes") and "vars" in h_data.axes.name:
    h_data_base = h_data[{"vars": 0}]
    h_mc_base = {proc: h[{"vars": 0}] for proc, h in h_mc_dict.items()}
else:
    h_data_base = h_data
    h_mc_base = h_mc_dict

# -------------------
# Build nonprompt seed template
# -------------------
h_nonprompt_base = None

if args.addNonprompt:
    h_nonprompt_base = make_nonprompt_template(
        h_data_base,
        h_mc_base,
        args.nonpromptFloor,
    )

# ----------------------------------
# write rabbit tensor
# ----------------------------------
writer = tensorwriter.TensorWriter(
    sparse=args.sparse,
    systematic_type=args.systematicType,
)

channel_name = "ch0"

writer.add_channel(h_data_base.axes, channel_name)
writer.add_data(h_data_base, channel_name)

# add the signal-like processes first, then backgrounds.
for proc in signal_procs:
    if proc in h_mc_base:
        writer.add_process(h_mc_base[proc], proc, channel_name, signal=False)

for proc in background_procs:
    if proc in h_mc_base:
        writer.add_process(h_mc_base[proc], proc, channel_name, signal=False)

if h_nonprompt_base is not None:
    writer.add_process(
        h_nonprompt_base,
        args.nonpromptName,
        channel_name,
        signal=False,
    )
    print_flush(f"Added synthetic nonprompt process: {args.nonpromptName}")

# -------------------------------------
# PDF systematics from *_pdfvars*_Corr
# -------------------------------------
for proc_name in signal_procs:
    if proc_name not in h_mc_base:
        continue

    if proc_name not in h_pdfvars_corr_dict:
        print_flush(f"Warning: no pdfvars _Corr histogram loaded for {proc_name}")
        continue

    h_pdfvars = h_pdfvars_corr_dict[proc_name]

    if "vars" not in h_pdfvars.axes.name:
        print_flush(
            f"Warning: pdfvars histogram for {proc_name} has no vars axis. "
            f"Axes: {[ax.name for ax in h_pdfvars.axes]}"
        )
        continue

    labels = vars_labels(h_pdfvars)
    print_flush(f"{proc_name} pdfvars labels: {labels[:8]}... total={len(labels)}")

    pdf_pairs = pdf_pairs_from_vars(labels)

    if not pdf_pairs:
        print_flush(f"Warning: no PDF variation pairs found for {proc_name}")
        continue

    n_added = 0

    for var_up, var_down, syst_name in pdf_pairs:
        h_up = h_pdfvars[{"vars": var_up}]
        h_down = h_pdfvars[{"vars": var_down}]

        h_up = project_to_nominal_axes(h_up, h_mc_base[proc_name])
        h_down = project_to_nominal_axes(h_down, h_mc_base[proc_name])

        writer.add_systematic(
            [h_up, h_down],
            syst_name,
            proc_name,
            channel_name,
            constrained=True,
            groups=["pdfs"],
        )

        n_added += 1

    print_flush(f"Added {n_added} PDF systematic pairs for {proc_name}")

# --------------------------------------
# alphaS systematic from *_pdfas*_Corr
# --------------------------------------
for proc_name in signal_procs:
    if proc_name not in h_mc_base:
        continue

    if proc_name not in h_pdfas_corr_dict:
        print_flush(f"Warning: no pdfas _Corr histogram loaded for {proc_name}")
        continue

    h_pdfas = h_pdfas_corr_dict[proc_name]

    if "vars" not in h_pdfas.axes.name:
        print_flush(
            f"Warning: pdfas histogram for {proc_name} has no vars axis. "
            f"Axes: {[ax.name for ax in h_pdfas.axes]}"
        )
        continue

    labels = vars_labels(h_pdfas)
    print_flush(f"{proc_name} pdfas labels: {labels}")

    # Preferred Z-like labels, with a fallback for older W test files.
    var_up = choose_existing_label(
        labels,
        [args.alphaSUpName, "alphaSUp"],
    )

    var_down = choose_existing_label(
        labels,
        [args.alphaSDownName, "alphaSDown"],
    )

    if var_up is None or var_down is None:
        print_flush(
            f"Warning: could not find both alphaS variations for {proc_name}. "
            f"Available labels: {labels}"
        )
        continue

    h_up = h_pdfas[{"vars": var_up}]
    h_down = h_pdfas[{"vars": var_down}]

    h_up = project_to_nominal_axes(h_up, h_mc_base[proc_name])
    h_down = project_to_nominal_axes(h_down, h_mc_base[proc_name])

    writer.add_systematic(
        [h_up, h_down],
        "pdfAlphaS",
        proc_name,
        channel_name,
        constrained=False,
        noi=True,
        groups=["alphaS"],
    )

    print_flush(
        f"Added pdfAlphaS systematic for {proc_name}: "
        f"up={var_up}, down={var_down}"
    )

# -----------------
# Write output
# -----------------
os.makedirs(args.output, exist_ok=True)

writer.write(
    outfolder=args.output,
    outfilename=args.outname,
)

print_flush(f"Tensor written to: {args.output}/{args.outname}.hdf5")

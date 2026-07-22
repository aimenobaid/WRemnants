"""Dump per-region smoothing polynomial coefficients for SmoothABCD.

Refit the linearized smoothing polynomial from a nominal fake histogram and
save the per-region polynomial coefficients in the SmoothdABCD
power-series basis.  The output can be loaded as initial parameter values in a
rabbit fit via ``params:PATH`` in the ``--paramModel SmoothABCD``
CLI token.

Typical use::
    python scripts/rabbit/regen_smoothing_params.py \\
        -i mw_5TeV.hdf5 \\
        -o /path/to/params.hdf5
"""

import os
import copy
import h5py
import numpy as np

from wremnants.postprocessing.datagroups.datagroups import Datagroups
from wremnants.postprocessing.histselections import FakeSelectorSimpleABCD
from wremnants.postprocessing.regression import Regressor
from wremnants.utilities.io_tools import base_io
from wremnants.utilities import common, parsing
from wums import ioutils, logging, output_tools

logger = logging.child_logger(__name__)

def make_parser():
    parser = parsing.base_parser()
    parser.description = __doc__
    parser.add_argument("-i","--inputFile",required=True,type=str, help="Input HDF5 histogram file (output of a histmaker).")
    parser.add_argument("-o","--outpath",required=True,type=str,help="Output path for the params HDF5 file (.hdf5 appended if missing).")
    parser.add_argument("--inputBaseName",default="nominal",type=str,help="Name of the nominal histogram inside the input file.")
    parser.add_argument("--fakerateAxes",nargs="+",default=["eta", "pt", "charge"],help="Axes for the fakerate binning.")
    parser.add_argument("--fakeEstimation",type=str,default="extended1D",choices=["simple", "extrapolate", "extended1D", "extended2D"],
        help="Fake estimation mode (must match what will be used in setupRabbit).")
    parser.add_argument("--fakeSmoothingMode",type=str,default="full",choices=FakeSelectorSimpleABCD.smoothing_modes,
        help="Smoothing mode for fake estimate.")
    parser.add_argument("--fakeSmoothingOrder",type=int,default=3, help="Polynomial order for the spectrum smoothing.")
    parser.add_argument("--fakeSmoothingPolynomial",type=str,default="chebyshev",choices=Regressor.polynomials,
        help="Polynomial type for the spectrum smoothing.")
    parser.add_argument("--lumiScale",type=float,default=1.0,
        help="Rescale equivalent luminosity by this value (must match the value used in carrot).")
    parser.add_argument("--excludeProcGroups",type=str,nargs="*",default=["QCD"],
        help="Process groups to exclude when building Datagroups.")
    parser.add_argument("--filterProcGroups",type=str,nargs="*",default=None,
            help="If set, keep only these process groups when building Datagroups.")
    parser.add_argument("--dataProcFilter",default="SingleMuon",type=str,help="Substring used to find the data process.")
    parser.add_argument("--promptProcFilters",nargs="+",default=["Wplusmunu","Wminusmunu","Wplustaunu","Wminustaunu","Zmumu","Ztautau"],
        help="Prompt MC processes subtracted from data to build the fake histogram.")
    parser.add_argument("--fakeFloor",type=float,default=1e-6,
        help="Minimum value used only when building smoothing initialization parameters.")
    parser.add_argument("--smoothingAxis", default="pt", type=str, help="Axis to smooth over.")
    parser.add_argument("--selectAxisBin",nargs=2,action="append",default=[],metavar=("AXIS", "BIN"),
        help="Select one bin of an axis before fitting smoothing params.")
    return parser

# -- helpers ---

def get_hist(results, proc, hist_name):
    h_proxy = results[proc]["output"][hist_name]
    return h_proxy.get() if hasattr(h_proxy, "get") else h_proxy

def has_hist(results, proc, hist_name):
    return (
        proc in results
        and "output" in results[proc]
        and hist_name in results[proc]["output"]
    )

def find_one_process(processes, filt):
    matches = [p for p in processes if filt in p]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one process matching '{filt}', found {matches}"
        )

    return matches[0]

def apply_axis_bin_selections(h, selections):
    h_out = h

    for axis_name, bin_idx in selections:
        bin_idx = int(bin_idx)

        if axis_name not in h_out.axes.name:
            raise RuntimeError(
                f"Cannot select {axis_name}:{bin_idx}; available axes are {h_out.axes.name}"
            )

        h_out = h_out[{axis_name: bin_idx}]

    return h_out

def make_fake_hist_from_file(inputFile, inputBaseName, dataProcFilter, promptProcFilters, fakeFloor):
    with h5py.File(inputFile, "r") as f:
        results = base_io.load_results_h5py(f)

        all_procs = [p for p in results.keys() if p != "meta_info"]
        data_proc = find_one_process(all_procs, dataProcFilter)

        if not has_hist(results, data_proc, inputBaseName):
            raise RuntimeError(
                f"Data process {data_proc} does not have histogram {inputBaseName}"
            )

        h_fakes = get_hist(results, data_proc, inputBaseName).copy()
        view_fake = h_fakes.view(flow=False)

        if not hasattr(view_fake, "value"):
            raise RuntimeError("Expected weighted histogram storage with .value/.variance.")

        for proc in all_procs:
            if proc == data_proc:
                continue

            if not any(filt in proc for filt in promptProcFilters):
                continue

            if not has_hist(results, proc, inputBaseName):
                continue

            h_prompt = get_hist(results, proc, inputBaseName)

            if h_prompt.axes.name != h_fakes.axes.name:
                raise RuntimeError(
                    f"Axis mismatch while building fake histogram for {proc}: "
                    f"data axes={h_fakes.axes.name}, prompt axes={h_prompt.axes.name}"
                )

            print(f"Subtracting prompt MC from fake seed: {proc}")
            view_prompt = h_prompt.view(flow=False)
            view_fake.value[...] = view_fake.value - view_prompt.value

            if view_fake.variance is not None and view_prompt.variance is not None:
                view_fake.variance[...] = view_fake.variance + view_prompt.variance

        mask = view_fake.value <= fakeFloor
        view_fake.value[mask] = fakeFloor

        if view_fake.variance is not None:
            view_fake.variance[mask] = fakeFloor

        total = h_fakes.sum().value if hasattr(h_fakes.sum(), "value") else h_fakes.sum()
        print(f"Built fake smoothing seed with total yield = {total}")

        return h_fakes

def make_chebyshev_matrix(axis, order):
    centers = np.array(axis.centers)
    edges = np.array(axis.edges)
    widths = np.array(axis.widths)

    x_cheby = 2.0 * (centers - edges[0]) / (edges[-1] - edges[0]) - 1.0

    T = np.zeros((len(x_cheby), order + 1))
    T[:, 0] = 1.0

    if order >= 1:
        T[:, 1] = x_cheby

    for k in range(2, order + 1):
        T[:, k] = 2.0 * x_cheby * T[:, k - 1] - T[:, k - 2]

    return T, widths

def fit_log_yield_coeffs(yields, T, widths, fakeFloor):
    if yields is None:
        raise RuntimeError("fit_log_yield_coeffs received None. Check get_region_yields return value.")

    y = np.asarray(yields, dtype=float)

    if y.shape != widths.shape:
        raise RuntimeError(f"Yield shape {y.shape} does not match smoothing-axis widths shape {widths.shape}")

    y = np.maximum(y, fakeFloor)

    if not np.all(np.isfinite(y)):
        raise RuntimeError("Non-finite fake yields found before log fit.")

    log_rate = np.log(y / widths)
    coeff_rate = np.linalg.lstsq(T, log_rate, rcond=None)[0]

    log_bw_coeffs = np.linalg.lstsq(T, np.log(widths), rcond=None)[0]

    coeffs = coeff_rate + log_bw_coeffs

    if not np.all(np.isfinite(coeffs)):
        raise RuntimeError("Non-finite smoothing coefficients produced.")

    return coeffs

def get_region_yields(values, axes, smoothingAxis, outerAxes, outerBins, dxyAxis, relIsoAxis, dxyBin, relIsoBin):
    slicer = [slice(None)] * len(axes)

    for axis_name, axis_bin in zip(outerAxes, outerBins):
        slicer[axes.index(axis_name)] = axis_bin

    slicer[axes.index(dxyAxis)] = dxyBin
    slicer[axes.index(relIsoAxis)] = relIsoBin

    y = values[tuple(slicer)]

    if y.ndim != 1:
        raise RuntimeError
    return y 
# ----------------------------------

def dump_smoothing_params(
    outpath,inputBaseName,inputFile,dataProcFilter,promptProcFilters,fakeFloor,
    smoothingAxis,dxyAxis,relIsoAxis,order,axisSelections=None,meta_data_dict=None,postfix="",
):
    """
    Dump the per-region Chebyshev polynomial coefficients of the nominal fake histogram smoothing fit, 
    in the layout expected by SmoothExtendedABCD as``initial_params``.

    Both the WRemnants spectrum regressor and SmoothExtendedABCD now use the 
    same Chebyshev basis (T_k of the first kind, with x̃ ∈ [-1, 1] via the axis edges).
    The regressor coefficients are passed through directly, with a projection of ``log(bin_width)`` added so the exported coefficients
    predict yields per bin, not rates per unit of the smoothing axis (the regressor fits ``log(data_rate) = log(data_yield / bin_width)``). 
    In the simultaneous ABCD fit the nonprompt process is filled with ``OnesSelector``, so ``mc_template = 1`` 
    and the polynomial must carry the full absolute scale --> the Chebyshev intercept (T_0) is kept.

    The saved array has shape (5 * n_outer * (order+1),) with the layout [A_params, B_params, C_params].

    Flat ABCD index ordering for the 3 regions with signal_region=False
    (FakeSelector1DABCD, with flow=True, after y-axis flip so tight iso is last):
        0=A (high relIso, high dxy), 1=B (low relIso, high dxy),
        2=C  (high relIso, low dxy) <-- application region?
    Note: signal_region=False drops flat index 3 = D (low relIso, low dxy = signal region), which is rabbit's predicted region.

    Histselections ↔ SmoothABCD model name mapping:
        histsel "application region" (high relIso + low dxy) = C_model (free)
        histsel "signal region"      (low relIso + low dxy) = D_model (predicted)

    Mapping to model order [A=0, B=1, C=2]:
        model_A  ← sideband flat 2
        model_B  ← sideband flat 3
        model_C  ← sideband flat 4
    """

    h_fakes = make_fake_hist_from_file(
        inputFile,
        inputBaseName,
        dataProcFilter,
        promptProcFilters,
        fakeFloor,
    )

    if axisSelections:
        print(f"Applying axis-bin selections before smoothing params: {axisSelections}")
        h_fakes = apply_axis_bin_selections(h_fakes, axisSelections)
        print(f"Fake histogram axes after selection: {list(h_fakes.axes.name)}")

    axes = list(h_fakes.axes.name)

    for axis_name in [smoothingAxis, dxyAxis, relIsoAxis]:
        if axis_name not in axes:
            raise RuntimeError(f"Axis {axis_name} not found in histogram axes {axes}")

    outer_axes = [
        ax for ax in axes
        if ax not in [smoothingAxis, dxyAxis, relIsoAxis]
    ]

    outer_shape = tuple(len(h_fakes.axes[ax]) for ax in outer_axes)
    n_outer_flat = int(np.prod(outer_shape)) if outer_shape else 1

    smooth_ax = h_fakes.axes[smoothingAxis]
    T, bin_widths = make_chebyshev_matrix(smooth_ax, order)

    values = h_fakes.view(flow=False).value

    params_model = np.zeros((n_outer_flat, 3, order + 1))

    for iouter, outer_bins in enumerate(np.ndindex(*outer_shape)):
        # physical A = high dxy, high relIso
        y_A_phys = get_region_yields(
            values, axes, smoothingAxis, outer_axes, outer_bins,
            dxyAxis, relIsoAxis, 1, 1
        )

        # physical B = high dxy, low relIso
        y_B_phys = get_region_yields(
            values, axes, smoothingAxis, outer_axes, outer_bins,
            dxyAxis, relIsoAxis, 1, 0
        )

        # physical C = low dxy, high relIso
        y_C_phys = get_region_yields(
            values, axes, smoothingAxis, outer_axes, outer_bins,
            dxyAxis, relIsoAxis, 0, 1
        )

        # SmoothABCD model order:
        # A_model <- physical B
        # B_model <- physical A
        # C_model <- physical C
        params_model[iouter, 0, :] = fit_log_yield_coeffs(y_B_phys, T, bin_widths, fakeFloor)
        params_model[iouter, 1, :] = fit_log_yield_coeffs(y_A_phys, T, bin_widths, fakeFloor)
        params_model[iouter, 2, :] = fit_log_yield_coeffs(y_C_phys, T, bin_widths, fakeFloor)

    # Both bases now agree (Chebyshev T_k, x̃ ∈ [-1, 1] via the axis edges).
    # The regressor fits log(data_rate) = log(data_yield / bin_width); the model
    # evaluates yield = exp(poly) * mc. With mc = 1 (OnesSelector in the
    # simultaneous ABCD fit) the target coefficients are those of
    # log(data_yield) = log_rate + log(bin_width). The log(bin_width)
    # contribution is projected onto the Chebyshev basis and added.
    
    q_model = params_model

    # Flatten to model's layout [A_block, B_block, C_block]
    # Each block: n_outer × (order+1) in C-order
    params_out = q_model.transpose(1, 0, 2).reshape(-1)  # (3 * n_outer * (order+1),)
    if outpath and not os.path.isdir(outpath):
        os.makedirs(outpath)

    outfile = f"{outpath}/params"
    if postfix:
        outfile += f"_{postfix}"

    outfile += ".hdf5"

    with h5py.File(outfile, mode="w") as f:
        f.create_dataset("params", data=params_out)
        f.create_dataset("order", data=np.array(order))
        f.create_dataset(
            "smoothing_axis_name",
            data=np.array(smoothingAxis, dtype=h5py.string_dtype()))
        f.create_dataset("n_outer", data=np.array(n_outer_flat))
        f.create_dataset("outer_shape", data=np.array(outer_shape, dtype="int64"))
        if meta_data_dict is not None:
            ioutils.pickle_dump_h5py("meta", meta_data_dict, f)

    logger.info(
        f"Saved smoothing initial params to {outfile}  "
        f"(shape {params_out.shape}, n_outer={n_outer_flat}, n_abcd=3, order={order})"
    )

def main():
    parser = make_parser()
    args = parser.parse_args()
    logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    meta_data_dict = {
        "meta_info": output_tools.make_meta_info_dict(
            args=args,
            wd=common.base_dir,
        ),
    }

    dump_smoothing_params(
        args.outpath,
        args.inputBaseName,
        args.inputFile,
        args.dataProcFilter,
        args.promptProcFilters,
        args.fakeFloor,
        args.smoothingAxis,
        "dxy",
        "relIso",
        args.fakeSmoothingOrder,
        axisSelections=args.selectAxisBin,
        meta_data_dict=meta_data_dict,
    )

if __name__ == "__main__":
    main()

import argparse
import h5py
import numpy as np

from wremnants.utilities.io_tools import base_io


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
        raise RuntimeError(f"Expected one process matching {filt}, found {matches}")

    return matches[0]


def apply_selection(h, selections):
    h_out = h

    for sel in selections:
        axis_name, bin_idx = sel.split(":")
        bin_idx = int(bin_idx)

        if axis_name not in h_out.axes.name:
            raise RuntimeError(
                f"Axis {axis_name} not found. Available axes: {h_out.axes.name}"
            )

        h_out = h_out[{axis_name: bin_idx}]

    return h_out


def get_region_yield(h, dxy_bin, reliso_bin):
    if "dxy" not in h.axes.name:
        raise RuntimeError(f"No dxy axis found. Axes are {h.axes.name}")

    if "relIso" not in h.axes.name:
        raise RuntimeError(f"No relIso axis found. Axes are {h.axes.name}")

    h_region = h[{"dxy": dxy_bin, "relIso": reliso_bin}]

    total = h_region.sum()
    return total.value if hasattr(total, "value") else total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputFile", required=True)
    parser.add_argument("--histName", default="wpt_abcd")
    parser.add_argument("--dataProcFilter", default="SingleMuon")
    parser.add_argument(
        "--promptProcFilters",
        nargs="+",
        default=[
            "Wplusmunu",
            "Wminusmunu",
            "Wplustaunu",
            "Wminustaunu",
            "Zmumu",
            "Ztautau",
        ],
    )
    parser.add_argument(
        "--useDataMinusPrompt",
        action="store_true",
        help="Use data - prompt MC instead of raw data.",
    )
    parser.add_argument(
        "--select",
        nargs="*",
        default=[],
        help="Optional axis-bin selections, e.g. --select w_mt:0 charge:0 eta:24",
    )

    args = parser.parse_args()

    with h5py.File(args.inputFile, "r") as f:
        results = base_io.load_results_h5py(f)
        all_procs = [p for p in results.keys() if p != "meta_info"]

        data_proc = find_one_process(all_procs, args.dataProcFilter)

        if not has_hist(results, data_proc, args.histName):
            raise RuntimeError(f"{data_proc} does not have histogram {args.histName}")

        h = get_hist(results, data_proc, args.histName).copy()

        if args.useDataMinusPrompt:
            print("Using fake seed: data - prompt MC")

            view = h.view(flow=False)

            for proc in all_procs:
                if proc == data_proc:
                    continue

                if not any(filt in proc for filt in args.promptProcFilters):
                    continue

                if not has_hist(results, proc, args.histName):
                    continue

                h_prompt = get_hist(results, proc, args.histName)

                if h_prompt.axes.name != h.axes.name:
                    raise RuntimeError(
                        f"Axis mismatch for {proc}: "
                        f"data axes={h.axes.name}, prompt axes={h_prompt.axes.name}"
                    )

                print(f"Subtracting {proc}")
                view.value[...] -= h_prompt.view(flow=False).value

        else:
            print("Using raw data only")

    if args.select:
        print(f"Applying selections: {args.select}")
        h = apply_selection(h, args.select)

    print(f"Final axes: {list(h.axes.name)}")

    A = get_region_yield(h, dxy_bin=1, reliso_bin=1)
    B = get_region_yield(h, dxy_bin=1, reliso_bin=0)
    C = get_region_yield(h, dxy_bin=0, reliso_bin=1)
    D = get_region_yield(h, dxy_bin=0, reliso_bin=0)

    pred = B * C / A if A != 0 else np.nan
    ratio = pred / D if D != 0 else np.nan

    print("\n=== ABCD yields ===")
    print(f"A = high dxy, high relIso = {A}")
    print(f"B = high dxy, low  relIso = {B}")
    print(f"C = low  dxy, high relIso = {C}")
    print(f"D = low  dxy, low  relIso = {D}")

    print("\n=== ABCD prediction ===")
    print(f"B*C/A = {pred}")
    print(f"D     = {D}")
    print(f"(B*C/A) / D = {ratio}")
    print(f"D / (B*C/A) = {D / pred if pred != 0 else np.nan}")


if __name__ == "__main__":
    main()
import os
import math

from wremnants.utilities import common, parsing, samples
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)
parser.add_argument("--flavor", default="mu", choices=["mu"], help="Lepton flavor")

args = parser.parse_args()
print("analysis_label =", analysis_label)
print("era =", args.era)
print("dataPath =", args.dataPath)
print("flavor =", getattr(args, "flavor", None))

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

import hist
import narf

from wremnants.production import (
    generator_level_definitions,
    systematics,
    theory_corrections,

)

from wremnants.production.datasets.dataset_tools import getDatasets
from wremnants.production.histmaker_tools import (
    write_analysis_output, 
    aggregate_groups, 
    scale_to_data,
)
from wremnants.production import generator_level_definitions

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    base_path=args.dataPath,
    era=args.era,
)

# ===== Theory Corrections ======

# import lz4.frame
# import pickle

# theory_corrs = args.theoryCorr
# theory_corr_base = f"{common.data_dir}/TheoryCorrections/5020GeV"

# change (pull WMUNU5020GeV when availible)

# def load_corr_hist_5020(filename, proc, histname):
#     """5020 GeV pickles use ZMUMU5020GEV keys and legacy hist names."""
#     with lz4.frame.open(filename) as f:
#         corr = pickle.load(f)
#     key = histname.replace("scetlib_dyturbo_LatticeNP", "scetlib_dyturboLatticeNP")
#     key = key.replace("_minnlo_ratio", "__minnlo_ratio")
#     return corr["ZMUMU5020GEV"][key]

# theory_corrections.load_corr_hist = load_corr_hist_5020
# corr_helpers = theory_corrections.load_corr_helpers(
#     [d.name for d in datasets if d.name in samples.wprocs], 
#     theory_corrs,
#     base_dir=theory_corr_base,
# )

# ===== Histogram axes ======

axis_nLepton = hist.axis.Integer(0, 5, name="nLepton", underflow=False)

axis_mu_pt = hist.axis.Regular(60, 25, 150, name="mu_pt")
axis_mu_eta = hist.axis.Regular(48, -2.4, 2.4, name="mu_eta")
axis_abs_mu_eta = hist.axis.Variable(
    [0.0, 0.25, 0.5, 1.0, 1.5, 2.4],
    name="abs_mu_eta",underflow=False,overflow=True,
)
axis_mu_charge = hist.axis.Integer(-1, 2, name="mu_charge", underflow=False, overflow=False)
axis_phi = hist.axis.Regular(50, -math.pi, math.pi, circular=True, name="phi")

axis_w_mt = hist.axis.Regular(80, 0, 160, name="w_mt")

axis_met_pt = hist.axis.Regular(60,0,150, name="met_pt")
axis_met_phi = hist.axis.Regular(50, -math.pi, math.pi, circular=True, name="met_phi")

axis_w_pt = hist.axis.Regular(80, 0, 160, name="w_pt")

axis_prefire_tensor = hist.axis.Integer(0, 2, name="prefire_variation", underflow=False, overflow=False)

def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    # ----- Event Weights -----
    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")
    df = df.Define("isEvenEvent", "event % 2 == 0")

    # ----- Event Selection: W -> mu nu (require 1 good mu and missing pT) ------

    df = df.Filter("HLT_HIMu17","Single-muon trigger")
    df = df.Define(
        "goodMu",
        "Muon_pt > 25 && abs(Muon_eta) < 2.4 && Muon_mediumId && Muon_isGlobal" 
    )  

    df = df.Define("goodMu_idx", "ROOT::VecOps::Nonzero(goodMu)")
    df = df.Filter("goodMu_idx.size() == 1", "Exactly one good muon")

    df = df.Filter("nElectron == 0", "No electrons in the event")

    # the one selected muon
    df = df.Define("i_mu", "int(goodMu_idx[0])") 
    # bookeeping: count reconstructed leptons
    df = df.Define("nLepton", "nElectron + nMuon") 

    # ------ MET kinematics ------
    # For now using stored NanoAOD MET directly. Later can replace with recoil-corrected MET from
    # recoilHelper.recoil_W.
    df = df.Alias("MET_corr_rec_pt", "MET_pt")
    df = df.Alias("MET_corr_rec_phi", "MET_phi")
    df = (
        df.Define("met_pt", "MET_corr_rec_pt")
        .Define("met_phi", "MET_corr_rec_phi")
    )
    # df = df.Filter("met_pt > 25", "MET requirement") # from CMS AN-25-085

    # ------- Muon kinematics -------
    MU_MASS = 0.105658

    df = (
        df.Define("mu_p4", f"ROOT::Math::PtEtaPhiMVector(Muon_pt[i_mu], Muon_eta[i_mu], Muon_phi[i_mu], {MU_MASS})")
          .Define("mu_pt", "Muon_pt[i_mu]")
          .Define("mu_eta", "Muon_eta[i_mu]")
          .Define("mu_phi", "Muon_phi[i_mu]")
          .Define("abs_mu_eta", "std::fabs(mu_eta)")
          # splitting charge (mostly d ubar --> W- and u dbar --> W+)
          .Define("mu_charge", "Muon_charge[i_mu]")
    )

    # # ----- Not used in current single-muon obv // W transverse obs, not needed for now ------
    df = (
        df.Define(
            "dphi_mu_met",
            "std::atan2(std::sin(mu_phi - met_phi), std::cos(mu_phi - met_phi))",
        )
          .Define(
              "w_mt",
              "std::sqrt(2.0 * mu_pt * met_pt * (1.0 - std::cos(dphi_mu_met)))",
          )
          .Define("mu_px", "mu_pt * std::cos(mu_phi)")
          .Define("mu_py", "mu_pt * std::sin(mu_phi)")
          .Define("met_px", "met_pt * std::cos(met_phi)")
          .Define("met_py", "met_pt * std::sin(met_phi)")
          .Define("w_px", "mu_px + met_px")
          .Define("w_py", "mu_py + met_py")
          .Define("w_pt", "std::sqrt(w_px*w_px + w_py*w_py)")
          .Define("w_phi", "std::atan2(w_py, w_px)")
    )

    #  half the boson weight
    df = df.Filter("w_mt > 40", "W transverse mass requirement")

    # add backgrounds =: Z/Y* , W --> taunu

    # prefiring
    if dataset.is_data:
        df = df.Define("nominal_weight", "1.0")
    else:
        df = df.Define("exp_weight", "weight*L1PreFiringWeight_Nom")
        df = generator_level_definitions.define_prefsr_vars(df)
        df = df.Alias("nominal_weight_uncorr", "exp_weight")
        df = df.Define("nominal_weight", "exp_weight")

        # df = df.DefinePerSample("central_pdf_weight", "1.0")
        # df = df.DefinePerSample("theory_weight_truncate", "10.0")
        # for theory_corr_name in theory_corrs:
        #     if theory_corr_name not in corr_helpers[dataset.name]:
        #         continue
        #     df = theory_corrections.define_theory_corr_weight_column(
        #         df, theory_corr_name
        #     )
        #     df = df.Define(
        #         f"{theory_corr_name}Weight_tensor",
        #         corr_helpers[dataset.name][theory_corr_name],
        #         [
        #             "massVgen",
        #             "absYVgen",
        #             "ptVgen",
        #             "chargeVgen",
        #             f"{theory_corr_name}_corr_weight",
        #         ],
        #    )

        # theory_corr_name = theory_corrs[0]
        # df = df.Define("nominal_weight", f"{theory_corr_name}Weight_tensor[0]")

    # ======= Histograms =======

    # ---- Fill histograms ----
    hist_nLepton = df.HistoBoost("nLepton", [axis_nLepton], ["nLepton", "nominal_weight"])
    #Muon
    hist_mu_pt = df.HistoBoost("mu_pt", [axis_mu_pt], ["mu_pt", "nominal_weight"])
    hist_mu_eta = df.HistoBoost("mu_eta", [axis_mu_eta], ["mu_eta", "nominal_weight"])
    hist_abs_mu_eta = df.HistoBoost("abs_mu_eta", [axis_abs_mu_eta], ["abs_mu_eta", "nominal_weight"])
    hist_mu_phi = df.HistoBoost("mu_phi", [axis_phi], ["mu_phi", "nominal_weight"])
    hist_mu_charge = df.HistoBoost("mu_charge", [axis_mu_charge], ["mu_charge", "nominal_weight"])

    #MET
    hist_met_pt = df.HistoBoost("met_pt", [axis_met_pt], ["met_pt", "nominal_weight"])
    hist_met_phi = df.HistoBoost("met_phi", [axis_phi], ["met_phi", "nominal_weight"])

    # W transverse observables
    hist_w_mt = df.HistoBoost("w_mt", [axis_w_mt], ["w_mt", "nominal_weight"])
    hist_w_pt = df.HistoBoost("w_pt", [axis_w_pt], ["w_pt", "nominal_weight"])
    hist_w_phi = df.HistoBoost("w_phi", [axis_phi], ["w_phi", "nominal_weight"])

    # ----- Charge-separated W obs -----

    df_plus = df.Filter("mu_charge > 0")
    df_minus = df.Filter("mu_charge < 0")
    # ---- un-rolled 2D histograms ----
    hist_mupt_absEta_plus = df_plus.HistoBoost("mupt_absEta_plus",[axis_mu_pt, axis_abs_mu_eta],["mu_pt", "abs_mu_eta", "nominal_weight"])
    hist_mupt_absEta_minus = df_minus.HistoBoost("mupt_absEta_minus",[axis_mu_pt, axis_abs_mu_eta],["mu_pt", "abs_mu_eta", "nominal_weight"])

    results += [
        hist_nLepton,
        hist_mu_pt,
        hist_mu_eta,
        hist_abs_mu_eta,
        hist_mu_phi,
        hist_mu_charge,
        hist_met_pt,
        hist_met_phi,
        hist_mupt_absEta_plus,
        hist_mupt_absEta_minus,
        hist_w_mt,
        hist_w_pt,
        hist_w_phi
    ]

    # === Prefiring variations ===

    if not dataset.is_data:
        df = df.Define(
            "prefire_vector",
            """
            auto res = std::vector<double>{
                L1PreFiringWeight_Muon_StatUp/L1PreFiringWeight_Muon_Nom,
                L1PreFiringWeight_Muon_StatDn/L1PreFiringWeight_Muon_Nom
            };

            res[0] = nominal_weight * res[0];
            res[1] = nominal_weight * res[1];

            return res;
            """,
        )

        df = df.Define(
            "prefire_vector_weight",
            "wrem::vec_to_tensor<2>(prefire_vector)",
        )

        hist_mueta_prefire = df.HistoBoost(
            "mueta_prefiring",
            [axis_mu_eta],
            ["mu_eta", "prefire_vector_weight"],
            tensor_axes=[axis_prefire_tensor],
        )

        hist_mupt_absEta_prefire = df.HistoBoost(
            "mupt_absEta_prefiring",
            [axis_mu_pt, axis_abs_mu_eta],
            ["mu_pt", "abs_mu_eta", "prefire_vector_weight"],
            tensor_axes=[axis_prefire_tensor],
        )

        hist_wpt_prefire = df.HistoBoost(
            "w_pt_prefiring",
            [axis_w_pt],
            ["w_pt", "prefire_vector_weight"],
            tensor_axes=[axis_prefire_tensor],
        )

        results += [
            hist_mueta_prefire,
            hist_mupt_absEta_prefire,
        ]

    return results, weightsum

logger.debug(f"Datasets are {[d.name for d in datasets]}")
resultdict = narf.build_and_run(datasets[::-1], build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
write_analysis_output(resultdict, fout, args)
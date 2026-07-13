import os
import math

from wremnants.utilities import common, parsing, samples, theory_utils
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)
parser.add_argument("--flavor", default="mu", choices=["mu"], help="Lepton flavor")
parser = parsing.set_parser_default(parser, "pdfs", ["ct18z"])

args = parser.parse_args()
print("analysis_label =", analysis_label)
print("era =", args.era)
print("flavor =", getattr(args, "flavor", None))

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

import hist
import narf

from wremnants.production import (
    generator_level_definitions,
    helicity_utils,
    systematics,
    theory_corrections,
    unfolding_tools
)

from wremnants.production.datasets.dataset_tools import getDatasets
from wremnants.production.histmaker_tools import (
    write_analysis_output, 
    aggregate_groups, 
    scale_to_data,
)

# ====================== 
# Dataset loading 
# ======================
datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    base_path=args.dataPath,
    era=args.era,
)

# ===================
# Theory Corrections 
# ===================

# import lz4.frame
# import pickle

# theory_corrs = args.theoryCorr
# theory_corr_base = f"{common.data_dir}/TheoryCorrections/5020GeV"

# change (pull WMUNU5020GeV when availible)

# def load_corr_hist_5020(filename, proc, histname):
#     """5020 GeV pickles use ZMUMU5020GEV keys and legacy hist names."""
    # with lz4.frame.open(filename) as f:
    #     corr = pickle.load(f)
#     key = histname.replace("scetlib_dyturbo_LatticeNP", "scetlib_dyturboLatticeNP")
    # key = key.replace("_minnlo_ratio", "__minnlo_ratio")
#     return corr["ZMUMU5020GEV"][key]

# theory_corrections.load_corr_hist = load_corr_hist_5020
# corr_helpers = theory_corrections.load_corr_helpers(
#     [d.name for d in datasets if d.name in samples.wprocs], 
#     theory_corrs,
#     base_dir=theory_corr_base,
# )

# ------- MiNNLO / CT18Z alphaS weight information --------
procs_v = [d.name for d in datasets if d.name in samples.vprocs]
theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
corr_helpers = theory_corrections.load_corr_helpers(procs_v, theory_corrs)
helicity_smoothing_helpers = {}

print("Theory correction processes:", procs_v)
print("Theory corrections:", theory_corrs)

# ---- PDF weight information -----
ct18z_pdf_info = theory_corrections.make_theory_corr_weight_info(
    "ct18z",alphas=False,renorm=True)
ct18z_pdf_weights = ct18z_pdf_info["weights"]
ct18z_pdf_labels = theory_utils.pdfNamesAsymHessian(len(ct18z_pdf_weights),"pdfCT18Z")

print("Number of CT18Z PDF weights:", len(ct18z_pdf_weights))
print("First few CT18Z PDF labels:", ct18z_pdf_labels[:6])

# ===================
#  Histogram axes 
# ===================
axis_nLepton = hist.axis.Integer(0, 5, name="nLepton", underflow=False)
axis_phi = hist.axis.Regular(50, -math.pi, math.pi, circular=True, name="phi")
eta_bins = [-2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0,
            0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]

axis_mu_pt = hist.axis.Regular(26, 18, 44, name="mu_pt")
axis_mu_eta = hist.axis.Variable(eta_bins, name="mu_eta", underflow=False, overflow=False)
axis_mu_charge = hist.axis.Integer(-2, 2, name="mu_charge", underflow=False, overflow=False)

axis_mu_relIso = hist.axis.Regular(80, 0.0, 1.0,name="mu_relIso",underflow=False,overflow=True)
axis_mu_abs_dxy = hist.axis.Regular(80, 0.0, 0.05,name="mu_abs_dxy",underflow=False,overflow=True)

axis_met_pt = hist.axis.Regular(60,0,150, name="met_pt")
axis_met_phi = hist.axis.Regular(50, -math.pi, math.pi, circular=True, name="met_phi")

axis_w_mt = hist.axis.Regular(80, 0, 160, name="w_mt")
axis_w_pt = hist.axis.Regular(80, 0, 160, name="w_pt")
axis_w_y = hist.axis.Regular(48, -2.4, 2.4, 
    name="w_y", underflow=False, overflow=True)

axis_prefire_tensor = hist.axis.Integer(0, 2, name="prefire_variation", underflow=False, overflow=False)
# making the W MiNNLO/CT18Z variation histograms look like the Z_Corr histograms.
axis_pdfas_vars = hist.axis.StrCategory(["central", "pdfCT18ZNNLO_as_0120", "pdfCT18ZNNLO_as_0116"],name="vars")
axis_pdfvars_vars = hist.axis.StrCategory(ct18z_pdf_labels,name="vars")

# --- diagnostic ---
axis_nu_disc_case = hist.axis.Regular(2, 0, 2, name="nu_disc_case", underflow=False, overflow=False)
axis_yW_compare = hist.axis.Regular(100, -5.0, 5.0, name="yW_compare", underflow=False, overflow=False)

# ------- ABCD axes -------
axis_abcd_pt = hist.axis.Regular(26, 18, 44,name="pt",underflow=False,overflow=False)
axis_abcd_eta = hist.axis.Regular(48, -2.4, 2.4,name="eta",underflow=False,overflow=False)
axis_abcd_charge = hist.axis.Regular(2, -2, 2,name="charge",underflow=False,overflow=False)
axis_abcd_dxy = hist.axis.Variable([0.0, 0.015, 1.0],name="dxy",underflow=False,overflow=False)
axis_abcd_relIso = hist.axis.Variable([0.0, 0.4, 10.0],name="relIso",underflow=False,overflow=False)
axis_wpt_abcd_pt = hist.axis.Regular(80, 0, 80, name="w_pt", underflow=False, overflow=False)

# =====================
# Main graph building 
# =====================

def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    # ------ Event Weights ------
    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")
    df = df.Define("isEvenEvent", "event % 2 == 0")

    # ------ Event Selection: W -> mu nu  -------
    df = df.Filter("HLT_HIMu17","Single-muon trigger")
    df = df.Define(
        "goodMu",
        "Muon_pt > 18 && abs(Muon_eta) < 2.4 && Muon_mediumId && Muon_isGlobal" 
    )  
    df = df.Define("goodMu_idx", "ROOT::VecOps::Nonzero(goodMu)")
    df = df.Filter("goodMu_idx.size() == 1", "Exactly one good muon")
    df = df.Filter("nElectron == 0", "No electrons in the event")
    df = df.Define("i_mu", "int(goodMu_idx[0])") 
    df = df.Define("nLepton", "nElectron + nMuon") 
    
    # ------ MET kinematics ------
    # For now using stored NanoAOD MET directly. 
    # can replace with recoil-corrected MET from recoilHelper.recoil_W later.
    # df = df.Alias("MET_corr_rec_pt", "MET_pt")
    # df = df.Alias("MET_corr_rec_phi", "MET_phi")

    df = (
        # df.Define("met_pt", "MET_corr_rec_pt")
        #   .Define("met_phi", "MET_corr_rec_phi")

        #   .Define("puppi_met_pt", "PuppiMET_pt")
        #   .Define("puppi_met_phi", "PuppiMET_phi")

        # Resolution deep met
        df.Define("deepmet_reso_pt", "DeepMETResolutionTune_pt") 
          .Define("deepmet_reso_phi", "DeepMETResolutionTune_phi")

        # # Response DEEP MET
        #   .Define("deepmet_resp_pt", "DeepMETResponseTune_pt")
        #   .Define("deepmet_resp_phi", "DeepMETResponseTune_phi")
    )
    # df = df.Filter("met_pt > 25", "MET requirement") 

    # ------- Muon kinematics -------
    MU_MASS = 0.105658
    df = (
        df.Define("mu_p4", f"ROOT::Math::PtEtaPhiMVector(Muon_pt[i_mu], Muon_eta[i_mu], Muon_phi[i_mu], {MU_MASS})")
          .Define("mu_pt", "Muon_pt[i_mu]")
          .Define("mu_eta", "Muon_eta[i_mu]")
          .Define("mu_phi", "Muon_phi[i_mu]")
          # splitting charge (mostly d ubar --> W- and u dbar --> W+)
          .Define("mu_charge", "Muon_charge[i_mu]")
          .Define("mu_relIso", "Muon_pfRelIso04_all[i_mu]") # relative isolation of muon
          .Define("mu_dxy", "Muon_dxy[i_mu]")
          .Define("mu_abs_dxy", "std::fabs(Muon_dxy[i_mu])") # dxy wrt first PV, in cm
          )
    df = (
        df.Define("abcd_pt", "mu_pt")
          .Define("abcd_eta", "mu_eta")
          .Define("abcd_charge", "double(mu_charge)")
          .Define("abcd_dxy", "mu_abs_dxy")
          .Define("abcd_relIso", "mu_relIso")
    )
    # ------- W transverse obs --------
    df = (
        df.Define("dphi_mu_met",
            "std::atan2(std::sin(mu_phi - deepmet_reso_phi), std::cos(mu_phi - deepmet_reso_phi))")
          .Define("w_mt",
              "std::sqrt(2.0 * mu_pt * deepmet_reso_pt * (1.0 - std::cos(dphi_mu_met)))")
          .Define("mu_px", "mu_pt * std::cos(mu_phi)")
          .Define("mu_py", "mu_pt * std::sin(mu_phi)")
          .Define("mu_pz", "mu_p4.Pz()")
          .Define("mu_E", "mu_p4.E()")
          .Define("met_px", "deepmet_reso_pt * std::cos(deepmet_reso_phi)")
          .Define("met_py", "deepmet_reso_pt * std::sin(deepmet_reso_phi)")
          .Define("w_px", "mu_px + met_px")
          .Define("w_py", "mu_py + met_py")
          .Define("w_pt", "std::sqrt(w_px*w_px + w_py*w_py)")
          .Define("w_phi", "std::atan2(w_py, w_px)")
          # to be fixed: 
          .Define("nu_disc",
            """
            const double mW = 80.379;
            const double ptl2 = mu_pt * mu_pt;
            const double A = mW*mW + 2.0*(mu_px*met_px + mu_py*met_py);
            return A*A - 4.0*ptl2*deepmet_reso_pt*deepmet_reso_pt;
            """)
          .Define("met_pt_for_w_y",
            """
            const double mW = 80.379;
            const double met = static_cast<double>(deepmet_reso_pt);
            if (nu_disc >= 0.0) {
                return met;
            }
            const double mt2 = 2.0 * mu_pt * met * (1.0 - std::cos(dphi_mu_met));
            if (mt2 <= 0.0) {
                return met;
            }
            const double k = (mW*mW) / mt2;
            return k * met;
            """)
          .Define("nu_pz",
            """
            const double mW = 80.379;
            const double ptl2 = mu_pt * mu_pt;

            if (ptl2 <= 0.0) {
                return 0.0;
            }

            const double A = mW*mW + 2.0*(mu_px*met_px + mu_py*met_py);
            const double disc = A*A - 4.0*ptl2*deepmet_reso_pt*deepmet_reso_pt;

            if (disc >= 0.0) {
                const double sqrt_disc = std::sqrt(disc);
                const double sol_plus  = (A*mu_pz + mu_E*sqrt_disc)/(2.0*ptl2);
                const double sol_minus = (A*mu_pz - mu_E*sqrt_disc)/(2.0*ptl2);
                if (std::fabs(sol_plus) < std::fabs(sol_minus)) {
                    return sol_plus;
                } else {
                    return sol_minus;
                }
            }

            // Negative-discriminant case:
            // rescale neutrino pT so that mT(W) = mW, giving one real pz solution.
            const double met_px_use = met_pt_for_w_y * std::cos(deepmet_reso_phi);
            const double met_py_use = met_pt_for_w_y * std::sin(deepmet_reso_phi);
            const double A_projected = mW*mW + 2.0*(mu_px*met_px_use + mu_py*met_py_use);
            return (A_projected * mu_pz)/(2.0*ptl2);
            """)
          .Define("nu_E",
              "std::sqrt(met_pt_for_w_y*met_pt_for_w_y + nu_pz*nu_pz)"
              )
          .Define("w_E", "mu_E + nu_E")
          .Define("w_pz", "mu_pz + nu_pz")
          .Define("w_y",
            """
            const double num = w_E + w_pz;
            const double den = w_E - w_pz;

            if (num <= 0.0 || den <= 0.0) {
                return -999.0;
            }

            return 0.5 * std::log(num/den);
            """)
          # diagnostic
          .Define("nu_disc_case", "nu_disc < 0.0 ? 0.5 : 1.5")
          
    )

    # df = df.Filter("w_mt > 40", "W transverse mass requirement")

    # --- nominal weight and alphaS tensor weights ---
    if dataset.is_data:
        df = df.Define("nominal_weight", "1.0")

    else:
        df = df.Define("exp_weight", "weight*L1PreFiringWeight_Nom")
        df = theory_corrections.define_theory_weights_and_corrs(
            df,dataset.name, corr_helpers, args, helicity_smoothing_helpers=helicity_smoothing_helpers,
        )
        # df = df.Define(
        #     "gen_w_y",
        #     """
        #     int best = -1;
        #     double best_pt = -1.0;
        #     for (int i = 0; i < nGenPart; ++i) {
        #         const bool isW = std::abs(GenPart_pdgId[i]) == 24;
        #         const bool isLastCopy = (GenPart_statusFlags[i] & (1 << 13));

        #         if (isW && isLastCopy && GenPart_pt[i] > best_pt) {
        #             best = i;
        #             best_pt = GenPart_pt[i];
        #         }
        #     }
        #     if (best < 0) {
        #         return -999.0;
        #     }
        #     ROOT::Math::PtEtaPhiMVector gen_w_p4(
        #         GenPart_pt[best],
        #         GenPart_eta[best],
        #         GenPart_phi[best],
        #         GenPart_mass[best]
        #     );
        #     const double E = gen_w_p4.E();
        #     const double pz = gen_w_p4.Pz();
        #     const double num = E + pz;
        #     const double den = E - pz;
        #     if (num <= 0.0 || den <= 0.0) {
        #         return -999.0;
        #     }
        #     return 0.5 * std::log(num / den);
        #     """
        # )

    # ------------------ ABCD cut -----------------------------------
    abcd_dxy_cut = 0.015
    abcd_relIso_cut = 0.4
    df_A = df.Filter(f"mu_abs_dxy >= {abcd_dxy_cut} && mu_relIso >= {abcd_relIso_cut}",
    "ABCD region A: high dxy, high relIso")
    df_B = df.Filter(
        f"mu_abs_dxy >= {abcd_dxy_cut} && mu_relIso < {abcd_relIso_cut}",
        "ABCD region B: high dxy, low relIso")
    df_C = df.Filter(
        f"mu_abs_dxy < {abcd_dxy_cut} && mu_relIso >= {abcd_relIso_cut}",
        "ABCD region C: low dxy, high relIso")
    df_D = df.Filter(
        f"mu_abs_dxy < {abcd_dxy_cut} && mu_relIso < {abcd_relIso_cut}",
        "ABCD region D/SR: low dxy, low relIso")

    # ===========
    # Histograms 
    # ===========

    # ===== Nominal Hists ====

    # ------ Event and muon kinematics -----
    hist_nLepton = df.HistoBoost("nLepton", [axis_nLepton], ["nLepton", "nominal_weight"])
    hist_mu_pt = df.HistoBoost("mu_pt", [axis_mu_pt], ["mu_pt", "nominal_weight"])
    hist_mu_eta = df.HistoBoost("mu_eta", [axis_mu_eta], ["mu_eta", "nominal_weight"])
    # hist_mu_phi = df.HistoBoost("mu_phi", [axis_phi], ["mu_phi", "nominal_weight"])
    hist_mu_charge = df.HistoBoost("mu_charge", [axis_mu_charge], ["mu_charge", "nominal_weight"])
    hist_mu_relIso = df.HistoBoost("mu_relIso",[axis_mu_relIso], ["mu_relIso", "nominal_weight"])
    hist_mu_dxy = df.HistoBoost("mu_abs_dxy",[axis_mu_abs_dxy],["mu_abs_dxy", "nominal_weight"])
    hist_mu_absdxy_relIso = df.HistoBoost("mu_absdxy_relIso",[axis_mu_abs_dxy, axis_mu_relIso],["mu_abs_dxy", "mu_relIso", "nominal_weight"])

    # ----- MET -----
    # hist_met_pt = df.HistoBoost("met_pt", [axis_met_pt], ["met_pt", "nominal_weight"])
    # hist_met_phi = df.HistoBoost("met_phi", [axis_phi], ["met_phi", "nominal_weight"])

    # ---- W transverse observables ----
    hist_w_mt = df.HistoBoost("w_mt", [axis_w_mt], ["w_mt", "nominal_weight"])
    hist_w_pt = df.HistoBoost("w_pt", [axis_w_pt], ["w_pt", "nominal_weight"])
    hist_w_phi = df.HistoBoost("w_phi", [axis_phi], ["w_phi", "nominal_weight"])

    # ------ ABCD cuts -------

    hist_w_pt_A = df_A.HistoBoost("w_pt_A",[axis_w_pt],["w_pt", "nominal_weight"])
    hist_w_pt_B = df_B.HistoBoost("w_pt_B",[axis_w_pt],["w_pt", "nominal_weight"])
    hist_w_pt_C = df_C.HistoBoost("w_pt_C",[axis_w_pt],["w_pt", "nominal_weight"])
    hist_w_pt_D = df_D.HistoBoost("w_pt_D",[axis_w_pt],["w_pt", "nominal_weight"])

    hist_mu_pt_A = df_A.HistoBoost("mu_pt_A",[axis_mu_pt],["mu_pt", "nominal_weight"])
    hist_mu_pt_B = df_B.HistoBoost("mu_pt_B",[axis_mu_pt],["mu_pt", "nominal_weight"])
    hist_mu_pt_C = df_C.HistoBoost("mu_pt_C",[axis_mu_pt],["mu_pt", "nominal_weight"])
    hist_mu_pt_D = df_D.HistoBoost("mu_pt_D",[axis_mu_pt],["mu_pt", "nominal_weight"])

    # hist_met_pt_A = df_A.HistoBoost("met_pt_A",[axis_met_pt],["met_pt", "nominal_weight"])
    # hist_met_pt_B = df_B.HistoBoost("met_pt_B",[axis_met_pt],["met_pt", "nominal_weight"])
    # hist_met_pt_C = df_C.HistoBoost("met_pt_C",[axis_met_pt],["met_pt", "nominal_weight"])
    # hist_met_pt_D = df_D.HistoBoost("met_pt_D",[axis_met_pt],["met_pt", "nominal_weight"])

    # hist_puppi_met_pt_A = df_A.HistoBoost("puppi_met_pt_A", [axis_met_pt], ["puppi_met_pt", "nominal_weight"])
    # hist_puppi_met_pt_B = df_B.HistoBoost("puppi_met_pt_B", [axis_met_pt], ["puppi_met_pt", "nominal_weight"])
    # hist_puppi_met_pt_C = df_C.HistoBoost("puppi_met_pt_C", [axis_met_pt], ["puppi_met_pt", "nominal_weight"])
    # hist_puppi_met_pt_D = df_D.HistoBoost("puppi_met_pt_D", [axis_met_pt], ["puppi_met_pt", "nominal_weight"])
    
    hist_deepmet_reso_pt_A = df_A.HistoBoost("deepmet_reso_pt_A", [axis_met_pt], ["deepmet_reso_pt", "nominal_weight"])
    hist_deepmet_reso_pt_B = df_B.HistoBoost("deepmet_reso_pt_B", [axis_met_pt], ["deepmet_reso_pt", "nominal_weight"])
    hist_deepmet_reso_pt_C = df_C.HistoBoost("deepmet_reso_pt_C", [axis_met_pt], ["deepmet_reso_pt", "nominal_weight"])
    hist_deepmet_reso_pt_D = df_D.HistoBoost("deepmet_reso_pt_D", [axis_met_pt], ["deepmet_reso_pt", "nominal_weight"])

    # hist_deepmet_resp_pt_A = df_A.HistoBoost("deepmet_resp_pt_A", [axis_met_pt], ["deepmet_resp_pt", "nominal_weight"])
    # hist_deepmet_resp_pt_B = df_B.HistoBoost("deepmet_resp_pt_B", [axis_met_pt], ["deepmet_resp_pt", "nominal_weight"])
    # hist_deepmet_resp_pt_C = df_C.HistoBoost("deepmet_resp_pt_C", [axis_met_pt], ["deepmet_resp_pt", "nominal_weight"])
    # hist_deepmet_resp_pt_D = df_D.HistoBoost("deepmet_resp_pt_D", [axis_met_pt], ["deepmet_resp_pt", "nominal_weight"])

    # ------ ABCD cuts -------
    abcd_axes = [axis_abcd_pt,axis_abcd_eta,axis_abcd_charge,axis_abcd_dxy,axis_abcd_relIso]
    abcd_cols = ["abcd_pt","abcd_eta","abcd_charge","abcd_dxy","abcd_relIso","nominal_weight"]
    hist_mu_abcd = df.HistoBoost("mu_abcd",abcd_axes, abcd_cols)

    wpt_abcd_axes = [axis_wpt_abcd_pt,axis_abcd_eta,axis_abcd_charge,axis_abcd_dxy,axis_abcd_relIso]
    wpt_abcd_cols = ["w_pt","abcd_eta","abcd_charge","abcd_dxy","abcd_relIso","nominal_weight"]
    hist_wpt_abcd = df.HistoBoost("wpt_abcd",wpt_abcd_axes,wpt_abcd_cols)

    # ----- Charge-separated W obs -----
    df_plus = df.Filter("mu_charge > 0")
    df_minus = df.Filter("mu_charge < 0")

    hist_w_pt_plus = df_plus.HistoBoost("w_pt_plus",[axis_w_pt], ["w_pt", "nominal_weight"])
    hist_w_pt_minus = df_minus.HistoBoost("w_pt_minus",[axis_w_pt],["w_pt", "nominal_weight"])

    # Unrolled 2D histograms.
    hist_mupt_eta_plus = df_plus.HistoBoost("mupt_eta_plus",[axis_mu_pt, axis_mu_eta],["mu_pt", "mu_eta", "nominal_weight"])
    hist_mupt_eta_minus = df_minus.HistoBoost("mupt_eta_minus",[axis_mu_pt, axis_mu_eta],["mu_pt", "mu_eta", "nominal_weight"])
    hist_wpt_mueta_plus = df_plus.HistoBoost("wpt_mueta_plus",[axis_w_pt, axis_mu_eta],["w_pt", "mu_eta", "nominal_weight"])
    hist_wpt_mueta_minus = df_minus.HistoBoost("wpt_mueta_minus",[axis_w_pt, axis_mu_eta],["w_pt", "mu_eta", "nominal_weight"])

    hist_wpt_y_plus = df_plus.HistoBoost("wpt_y_plus",[axis_w_pt, axis_w_y],["w_pt", "w_y", "nominal_weight"])
    hist_wpt_y_minus = df_minus.HistoBoost("wpt_y_minus",[axis_w_pt, axis_w_y],["w_pt", "w_y", "nominal_weight"])
    # ------ diagnostic hists ----
    hist_nu_disc_case = df.HistoBoost("nu_disc_case",[axis_nu_disc_case],["nu_disc_case","nominal_weight"])


    results += [
        hist_nLepton,
        hist_mu_pt,
        hist_mu_eta,
        hist_mu_charge,
        # hist_met_pt,
        hist_mupt_eta_plus,
        hist_mupt_eta_minus,
        hist_w_mt,
        hist_w_pt,
        hist_w_pt_plus,
        hist_w_pt_minus,\
        
        hist_wpt_mueta_plus,
        hist_wpt_mueta_minus,

        # hist_mu_phi,
        # hist_met_phi,
        # hist_w_phi,

        # hist_wpt_y_minus,
        # hist_wpt_y_plus,

        hist_mu_relIso,
        hist_mu_dxy,
        hist_mu_absdxy_relIso,

        # hist_nu_disc_case,
        hist_w_pt_A,
        hist_w_pt_B,
        hist_w_pt_C,
        hist_w_pt_D,

        hist_mu_pt_A, 
        hist_mu_pt_B,
        hist_mu_pt_C,
        hist_mu_pt_D,

        # hist_met_pt_A, 
        # hist_met_pt_B,
        # hist_met_pt_C,
        # hist_met_pt_D,

        # hist_puppi_met_pt_A,
        # hist_puppi_met_pt_B,
        # hist_puppi_met_pt_C,
        # hist_puppi_met_pt_D,

        hist_deepmet_reso_pt_A,
        hist_deepmet_reso_pt_B,
        hist_deepmet_reso_pt_C,
        hist_deepmet_reso_pt_D,

        # hist_deepmet_resp_pt_A,
        # hist_deepmet_resp_pt_B,
        # hist_deepmet_resp_pt_C,
        # hist_deepmet_resp_pt_D,

        hist_mu_abcd,
        hist_wpt_abcd,
    ]

    # ===== alphaS variation histograms (MC ONLY) ======
    if not dataset.is_data:
        hist_w_pt_pdfas_corr = df.HistoBoost("w_pt_minnlo_pdfas_Corr",[axis_w_pt],["w_pt", "pdfCT18ZASWeights_tensor"],
            tensor_axes=[axis_pdfas_vars])
        hist_w_pt_plus_pdfas_corr = df_plus.HistoBoost("w_pt_plus_minnlo_pdfas_Corr", [axis_w_pt], ["w_pt", "pdfCT18ZASWeights_tensor"],
            tensor_axes=[axis_pdfas_vars])
        hist_w_pt_minus_pdfas_corr = df_minus.HistoBoost("w_pt_minus_minnlo_pdfas_Corr", [axis_w_pt], ["w_pt", "pdfCT18ZASWeights_tensor"],
            tensor_axes=[axis_pdfas_vars])
        hist_wpt_mueta_plus_pdfas_corr = df_plus.HistoBoost("wpt_mueta_plus_minnlo_pdfas_Corr",[axis_w_pt, axis_mu_eta], 
            ["w_pt", "mu_eta", "pdfCT18ZASWeights_tensor"],tensor_axes=[axis_pdfas_vars])
        hist_wpt_mueta_minus_pdfas_corr = df_minus.HistoBoost("wpt_mueta_minus_minnlo_pdfas_Corr",[axis_w_pt, axis_mu_eta], 
            ["w_pt", "mu_eta", "pdfCT18ZASWeights_tensor"],tensor_axes=[axis_pdfas_vars])
        hist_mu_abcd_pdfas_corr = df.HistoBoost("mu_abcd_minnlo_pdfas_Corr", abcd_axes,
                ["abcd_pt","abcd_eta","abcd_charge","abcd_dxy","abcd_relIso","pdfCT18ZASWeights_tensor"],tensor_axes=[axis_pdfas_vars])
        hist_mupt_eta_plus_pdfas_corr = df_plus.HistoBoost("mupt_eta_plus_minnlo_pdfas_Corr",[axis_mu_pt, axis_mu_eta],
            ["mu_pt", "mu_eta", "pdfCT18ZASWeights_tensor"],tensor_axes=[axis_pdfas_vars])
        hist_mupt_eta_minus_pdfas_corr = df_minus.HistoBoost("mupt_eta_minus_minnlo_pdfas_Corr",[axis_mu_pt, axis_mu_eta],
            ["mu_pt", "mu_eta", "pdfCT18ZASWeights_tensor"],tensor_axes=[axis_pdfas_vars])
        hist_wpt_abcd_pdfas_corr = df.HistoBoost("wpt_abcd_minnlo_pdfas_Corr",wpt_abcd_axes,
            ["w_pt","abcd_eta","abcd_charge","abcd_dxy","abcd_relIso","pdfCT18ZASWeights_tensor"],tensor_axes=[axis_pdfas_vars])
        
        results += [
            hist_w_pt_pdfas_corr,
            hist_w_pt_plus_pdfas_corr,
            hist_w_pt_minus_pdfas_corr,
            hist_wpt_mueta_plus_pdfas_corr,
            hist_wpt_mueta_minus_pdfas_corr,
            hist_mu_abcd_pdfas_corr,
            hist_wpt_abcd_pdfas_corr,
            hist_mupt_eta_plus_pdfas_corr,
            hist_mupt_eta_minus_pdfas_corr,
        ]

    # ====== pdf variation histograms (MC ONLY) ========
    if not dataset.is_data:
        hist_w_pt_pdfvars_corr = df.HistoBoost("w_pt_minnlo_pdfvars_Corr",[axis_w_pt],
            ["w_pt", "pdfCT18ZWeights_tensor"],tensor_axes=[axis_pdfvars_vars])
        hist_w_pt_plus_pdfvars_corr = df_plus.HistoBoost("w_pt_plus_minnlo_pdfvars_Corr",[axis_w_pt], ["w_pt",  "pdfCT18ZWeights_tensor"],
            tensor_axes=[axis_pdfvars_vars])
        hist_w_pt_minus_pdfvars_corr = df_minus.HistoBoost("w_pt_minus_minnlo_pdfvars_Corr",[axis_w_pt], ["w_pt",  "pdfCT18ZWeights_tensor"],
            tensor_axes=[axis_pdfvars_vars])
        hist_wpt_mueta_plus_pdfvars_corr = df_plus.HistoBoost("wpt_mueta_plus_minnlo_pdfvars_Corr",[axis_w_pt, axis_mu_eta],
            ["w_pt", "mu_eta", "pdfCT18ZWeights_tensor"],tensor_axes=[axis_pdfvars_vars])
        hist_wpt_mueta_minus_pdfvars_corr = df_minus.HistoBoost("wpt_mueta_minus_minnlo_pdfvars_Corr",[axis_w_pt, axis_mu_eta],
            ["w_pt", "mu_eta", "pdfCT18ZWeights_tensor"],tensor_axes=[axis_pdfvars_vars])
        hist_mu_abcd_pdfvars_corr = df.HistoBoost("mu_abcd_minnlo_pdfvars_Corr", abcd_axes, 
                ["abcd_pt","abcd_eta","abcd_charge","abcd_dxy","abcd_relIso","pdfCT18ZWeights_tensor"], tensor_axes=[axis_pdfvars_vars])
        hist_mupt_eta_plus_pdfvars_corr = df_plus.HistoBoost("mupt_eta_plus_minnlo_pdfvars_Corr",[axis_mu_pt, axis_mu_eta],
            ["mu_pt", "mu_eta", "pdfCT18ZWeights_tensor"],tensor_axes=[axis_pdfvars_vars])
        hist_mupt_eta_minus_pdfvars_corr = df_minus.HistoBoost("mupt_eta_minus_minnlo_pdfvars_Corr",[axis_mu_pt, axis_mu_eta],
            ["mu_pt", "mu_eta", "pdfCT18ZWeights_tensor"],tensor_axes=[axis_pdfvars_vars])
        hist_wpt_abcd_pdfvars_corr = df.HistoBoost("wpt_abcd_minnlo_pdfvars_Corr",wpt_abcd_axes,
            ["w_pt","abcd_eta","abcd_charge","abcd_dxy","abcd_relIso","pdfCT18ZWeights_tensor"],tensor_axes=[axis_pdfvars_vars])

        results += [
            hist_w_pt_pdfvars_corr,
            hist_w_pt_plus_pdfvars_corr,
            hist_w_pt_minus_pdfvars_corr,
            hist_wpt_mueta_plus_pdfvars_corr,
            hist_wpt_mueta_minus_pdfvars_corr,
            hist_mu_abcd_pdfvars_corr,
            hist_wpt_abcd_pdfvars_corr,
            hist_mupt_eta_plus_pdfvars_corr,
            hist_mupt_eta_minus_pdfvars_corr,
        ]

    # ===== gen comparison (MC ONLY) ========
    # if not dataset.is_data: 
    #     hist_w_y_reco_compare = df.HistoBoost("w_y_reco_compare",[axis_yW_compare],["w_y", "nominal_weight"])
    #     hist_gen_w_y_compare = df.HistoBoost("gen_w_y_compare",[axis_yW_compare],["gen_w_y", "nominal_weight"])

    #     results += [
    #         hist_w_y_reco_compare,
    #         hist_gen_w_y_compare,
    #     ]
    
    # ============= Prefiring variations ==================

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
        df = df.Define("prefire_vector_weight","wrem::vec_to_tensor<2>(prefire_vector)")
        hist_mueta_prefire = df.HistoBoost("mueta_prefiring",[axis_mu_eta],
            ["mu_eta", "prefire_vector_weight"],tensor_axes=[axis_prefire_tensor])
        hist_mupt_eta_prefire = df.HistoBoost("mupt_eta_prefiring",[axis_mu_pt, axis_mu_eta],
            ["mu_pt", "mu_eta", "prefire_vector_weight"],tensor_axes=[axis_prefire_tensor])
        hist_wpt_prefire = df.HistoBoost("w_pt_prefiring",[axis_w_pt],
            ["w_pt", "prefire_vector_weight"],tensor_axes=[axis_prefire_tensor])

        results += [
            hist_mueta_prefire,
            hist_mupt_eta_prefire,
            hist_wpt_prefire,
        ]

    return results, weightsum

# ===============================
# Run RDF graph and write output
# ===============================

logger.debug(f"Datasets are {[d.name for d in datasets]}")
resultdict = narf.build_and_run(datasets[::1], build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
write_analysis_output(resultdict, fout, args)
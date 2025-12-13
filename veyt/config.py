# Configuration file with all curve definitions
from datetime import date

FUTURES = {
    "eua": "cc_mp_ice-endex_eua-future_mc_ice_d_eur/t",
}

COMMODITIES = {
    "TTF": "gas_mp_ice-endex_eu_ttf-future_mc_ice_d_eur/mwh",
    "Coal": "coa_mp_ifeu_ara-api2-futures_mc_ice_d_usd/t",
    "German Power": "pwr_mp_eex_de_base-future_mc_eex_d_eur/mwh",
    "Brent Crude": "oil_mp_ifeu_brent-crude-futures_mc_ice_d_usd/bbl",
}

VOLATILITY = {
    "20d": "cc_price_a_eua-future-dec_mc_veyt_quant-future-volatility_20d-roll_d_d_eur/t",
    "5d": "cc_price_a_eua-future-dec_mc_veyt_quant-future-volatility_5d-roll_d_d_eur/t",
}

COT_ICE = {
    "IFCI_Positions": "cc_num-pos_eua-future_inv-firms-cred-inst_ice-cot_w_w_t",
    "IFCI_Holders": "cc_num-pers-hold-pos_eua-future_inv-firms-cred-inst_ice-cot_w_w_count",
    "IF_Positions": "cc_num-pos_eua-future_inv-funds_ice-cot_w_w_t",
    "IF_Holders": "cc_num-pers-hold-pos_eua-future_inv-funds_ice-cot_w_w_count",
    "OFI_Positions": "cc_num-pos_eua-future_other-inv-ints_ice-cot_w_w_t",
    "OFI_Holders": "cc_num-pers-hold-pos_eua-future_other-inv-ints_ice-cot_w_w_count",
    "CU_Positions": "cc_num-pos_eua-future_com-under_ice-cot_w_w_t",
    "CU_Holders": "cc_num-pers-hold-pos_eua-future_com-under_ice-cot_w_w_count",
    "OWCO_Positions": "cc_num-pos_eua-future_ops-w-compl-oblig_ice-cot_w_w_t",
    "OWCO_Holders": "cc_num-pers-hold-pos_eua-future_ops-w-compl-oblig_ice-cot_w_w_count",
}

COT_EEX = {
    "IFCI_Positions": "cc_num-pos_eua-future_inv-firms-cred-inst_eex-cot_w_w_t",
    "IFCI_Holders": "cc_num-pers-hold-pos_eua-future_inv-firms-cred-inst_eex-cot_w_w_count",
    "IF_Positions": "cc_num-pos_eua-future_inv-funds_eex-cot_w_w_t",
    "IF_Holders": "cc_num-pers-hold-pos_eua-future_inv-funds_eex-cot_w_w_count",
    "OFI_Positions": "cc_num-pos_eua-future_other-inv-ints_eex-cot_w_w_t",
    "OFI_Holders": "cc_num-pers-hold-pos_eua-future_other-inv-ints_eex-cot_w_w_count",
    "CU_Positions": "cc_num-pos_eua-future_com-under_eex-cot_w_w_t",
    "CU_Holders": "cc_num-pers-hold-pos_eua-future_com-under_eex-cot_w_w_count",
    "OWCO_Positions": "cc_num-pos_eua-future_ops-w-compl-oblig_eex-cot_w_w_t",
    "OWCO_Holders": "cc_num-pers-hold-pos_eua-future_ops-w-compl-oblig_eex-cot_w_w_count",
}

COT_TTF = {
    "IFCI_Positions": "cc_num-pos_ttf-future_inv-firms-cred-inst_ice-cot_w_w_mw",
    "IFCI_Holders": "cc_num-pers-hold-pos_ttf-future_inv-firms-cred-inst_ice-cot_w_w_count",
    "IF_Positions": "cc_num-pos_ttf-future_inv-funds_ice-cot_w_w_mw",
    "IF_Holders": "cc_num-pers-hold-pos_ttf-future_inv-funds_ice-cot_w_w_count",
    "OFI_Positions": "cc_num-pos_ttf-future_other-inv-ints_ice-cot_w_w_mw",
    "OFI_Holders": "cc_num-pers-hold-pos_ttf-future_other-inv-ints_ice-cot_w_w_count",
    "CU_Positions": "cc_num-pos_ttf-future_com-under_ice-cot_w_w_mw",
    "CU_Holders": "cc_num-pers-hold-pos_ttf-future_com-under_ice-cot_w_w_count",
    "OWCO_Positions": "cc_num-pos_ttf-future_ops-w-compl-oblig_ice-cot_w_w_mw",
    "OWCO_Holders": "cc_num-pers-hold-pos_ttf-future_ops-w-compl-oblig_ice-cot_w_w_count",
}

EU_AUCTIONS = {
    "Germany": "cc_ar_eex_de_eua-auction_eex_w_eur/t",
    "EU": "cc_ar_eex_eu_eua-auction_eex_3w_eur/t",
    "Poland": "cc_ar_eex_pl_eua-auction_eex_w2_eur/t",
}

WEATHER = {
    "cdd": "wthr_cdd_b_eea_veyt_ecmwf_quant-weather-ec_op_ec_avg-popul_h_h12_ch",
    "hdd": "wthr_hdd_b_eea_veyt_ecmwf_quant-weather-ec_op_ec_avg-popul_h_h12_ch",
}

SPREADS = {
    "clean_dark": "cc_spread-clean-dark-fm_a_veyt_quant-cspread_ice-eua-front-dec_medium-efficiency_d_d_eur/mwh",
    "clean_spark": "cc_spread-clean-spark-fm_a_veyt_quant-cspread_ice-eua-front-dec_medium-efficiency_d_d_eur/mwh",
    "fuel_switch": "cc_fuel-switch-price-front-month_a_veyt_quant-co2-fuel-switching-calc_d_d_eur/t",
}
OPTIONS = {
    "opt_put": "cc_opt-put_ice-endex_eua-future-opt_mc_ice_d_eur/t",
    "opt_call": "cc_opt-call_ice-endex_eua-future-opt_mc_ice_d_eur/t",
}
# Default date ranges
DEFAULT_FROM_DATE = "2021-01-01"
DEFAULT_UNTIL_DATE = date.today().isoformat()
gcloud_filepath = "gs://dashboard_data_ge"  # Set if uploading to google cloud server

DEFAULT_OUTPUT_DIR = f"{gcloud_filepath}/veyt_data_new"
# DEFAULT_OUTPUT_DIR = "veyt_data_new"

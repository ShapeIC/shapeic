import pandas as pd

df_shapeic = pd.read_csv("results.csv")

print(df_shapeic.columns)

#df_shapeic = df_shapeic.loc[df_shapeic["phase_margin_deg"]>60].copy()

print(
    df_shapeic.assign(
        dp_width_m=df_shapeic["diff_pair_width_m"] * 1e6,
        dp_length_m=df_shapeic["diff_pair_length_m"] * 1e6,
        dp_nf=df_shapeic["diff_pair_nf"],
        cm_width_m=df_shapeic["current_mirror_width_m"] * 1e6,
        cm_length_m=df_shapeic["current_mirror_length_m"] * 1e6,
        cm_nf=df_shapeic["current_mirror_nf"],
        cs_width_m=df_shapeic["common_source_width_m"] * 1e6,
        cs_length_m=df_shapeic["common_source_length_m"] * 1e6,
        cs_nf=df_shapeic["common_source_nf"],
        r_comp=df_shapeic["r_comp"],
        c_comp=df_shapeic["c_comp"],
        gain=df_shapeic["dc_gain_db"],
        f3db=df_shapeic["bandwidth_3db_hz"],
        pm  = df_shapeic["phase_margin_deg"]
    )[["dp_width_m", "dp_length_m", "dp_nf", "cm_width_m", "cm_length_m", "cm_nf", "cs_width_m", "cs_length_m", "cs_nf", "r_comp", "c_comp", "gain", "f3db", "pm"]].head(60)
)

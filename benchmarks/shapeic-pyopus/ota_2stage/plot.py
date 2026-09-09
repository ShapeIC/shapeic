import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from paretoset import paretoset

df_shapeic = pd.read_csv("shapeic/results.csv")

df_shapeic["area"] = 2*df_shapeic["diff_pair_width_m"]*df_shapeic["diff_pair_length_m"]+2*df_shapeic["current_mirror_width_m"]*df_shapeic["current_mirror_length_m"]+df_shapeic["common_source_width_m"]*df_shapeic["common_source_length_m"]
#df_shapeic = df_shapeic.sort_values(by="area")

df_shapeic_filtered = df_shapeic.loc[df_shapeic["area"]<1e-8].copy()

print(df_shapeic.columns)
print(
    df_shapeic.assign(
        area=df_shapeic["area"] * 1e12,
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
    )[["area", "dp_width_m", "dp_length_m", "dp_nf", "cm_width_m", "cm_length_m", "cm_nf", "cs_width_m", "cs_length_m", "cs_nf", "r_comp", "c_comp", "gain", "f3db", "pm"]].head(60)
)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.scatterplot(data=df_shapeic, x = "area", y = "dc_gain_db", ax=axes[0], hue="gain_1stage")
#sns.scatterplot(data=df_shapeic, x = "area", y = "bandwidth_3db_hz", ax=axes[0,1], hue="ro_ota_ohm")
#sns.scatterplot(data=df_shapeic, x = "area", y = "unity_gain_hz", ax=axes[1,0], hue="ro_ota_ohm")
sns.scatterplot(data=df_shapeic, x = "area", y = "phase_margin_deg", ax=axes[1], hue="gain_1stage")

axes[0].set_xlabel("Area [µm²]")
axes[0].set_ylabel("DC Gain [dB]")

axes[1].set_xlabel("Area [µm²]")
axes[1].set_ylabel("Phase Margin [°]")

#for ax in axes:
#    ax.set_xscale("log")
#    ax.grid(True, which="both")

axes[0].set_xscale("log")
#axes[0,1].set_xscale("log")
#axes[1,0].set_xscale("log")
axes[1].set_xscale("log")

#axes[0,1].set_yscale("log")

axes[0].grid(True, which="both")
#axes[0,1].grid(True, which="both")
#axes[1,0].grid(True, which="both")
axes[1].grid(True, which="both")

fig.savefig("shapeic_exploration.png", dpi=300, bbox_inches="tight")
plt.close(fig)


sns.scatterplot(data=df_shapeic, x = "area", y = "dc_gain_db", hue="gain_1stage")
plt.xlabel("Area [µm²]")
plt.ylabel("DC Gain [dB]")
plt.savefig("shapeic_exploration_onlygain.png", dpi=300, bbox_inches="tight")
plt.close()


pareto_mask = paretoset(
    df_shapeic_filtered[["area", "dc_gain_db", "bandwidth_3db_hz", "phase_margin_deg"]],
    sense=["min", "max", "max", "max"]    
)
df_pareto = df_shapeic_filtered[pareto_mask]
df_pyopus = pd.read_csv("pyopus/ota_opt_iterations.csv")

fig, ax = plt.subplots()
sns.scatterplot(data=df_shapeic, x = "area", y = "dc_gain_db", hue="gain_1stage", legend="brief")
sns.scatterplot(data=df_pyopus.iloc[[-1]], x = "area", y = "gain", marker="*", s=600, color="red", legend=False)
plt.xlabel("Area [µm²]")
plt.ylabel("DC Gain [dB]")
ax.legend(title="gain_1stage")
plt.savefig("shapeic_exploration_onlygain_with_pyopus.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.scatterplot(data=df_pareto, x = "area", y = "dc_gain_db", ax=axes[0,0])
sns.scatterplot(data=df_pyopus.iloc[[-1]], x = "area", y = "gain", ax=axes[0,0])
sns.scatterplot(data=df_pareto, x = "area", y = "bandwidth_3db_hz", ax=axes[0,1])
sns.scatterplot(data=df_pyopus.iloc[[-1]], x = "area", y = "f3db", ax=axes[0,1])
sns.scatterplot(data=df_pareto, x = "area", y = "phase_margin_deg", ax=axes[1,0])
sns.scatterplot(data=df_pyopus.iloc[[-1]], x = "area", y = "pm", ax=axes[1,0])
axes[0,0].set_xscale("log")
axes[0,1].set_xscale("log")
axes[1,0].set_xscale("log")

#x = df_pyopus["area"]
#y = df_pyopus["gain"]
#for i in range(len(df_pyopus) - 1):
#    ax.annotate(
#        "",
#        xy=(x.iloc[i + 1], y.iloc[i + 1]),
#        xytext=(x.iloc[i], y.iloc[i]),
#        arrowprops=dict(
#            arrowstyle="->",
#            lw=1.5,
#        ),
#    )
fig.savefig("sns_testplot.png", dpi=300, bbox_inches="tight")

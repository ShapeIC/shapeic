import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from paretoset import paretoset

df_shapeic = pd.read_csv("shapeic/results.csv")

df_shapeic["area"] = 2*df_shapeic["diff_pair_width_m"]*df_shapeic["diff_pair_length_m"]+2*df_shapeic["current_mirror_width_m"]*df_shapeic["current_mirror_length_m"]+df_shapeic["common_source_width_m"]*df_shapeic["common_source_length_m"]
df_shapeic = df_shapeic.sort_values(by="area")

df_shapeic_filtered = df_shapeic.loc[df_shapeic["area"]<1e-8].copy()

print(df_shapeic.columns)
print(
    df_shapeic.assign(
        area=df_shapeic["area"] * 1e12,
        dp_width_m=df_shapeic["diff_pair_width_m"] * 1e6,
        dp_length_m=df_shapeic["diff_pair_length_m"] * 1e6,
        cm_width_m=df_shapeic["current_mirror_width_m"] * 1e6,
        cm_length_m=df_shapeic["current_mirror_length_m"] * 1e6,
        cs_width_m=df_shapeic["common_source_width_m"] * 1e6,
        cs_length_m=df_shapeic["common_source_length_m"] * 1e6,
    )[["area", "dp_width_m", "dp_length_m", "cm_width_m", "cm_length_m", "cs_width_m", "cs_length_m"]].head(60)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(data=df_shapeic, x = "area", y = "dc_gain_db", ax=axes[0])
sns.scatterplot(data=df_shapeic, x = "area", y = "bandwidth_3db_hz", ax=axes[1])

for ax in axes:
    ax.set_xscale("log")
    ax.grid(True, which="both")

fig.savefig("shapeic_exploration.png", dpi=300, bbox_inches="tight")
plt.close(fig)

pareto_mask = paretoset(
    df_shapeic_filtered[["area", "dc_gain_db", "bandwidth_3db_hz"]],
    sense=["min", "max", "max"]    
)
df_pareto = df_shapeic_filtered[pareto_mask]
df_pyopus = pd.read_csv("pyopus/ota_opt_iterations.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=df_pareto, x = "area", y = "dc_gain_db", ax=axes[0])
sns.scatterplot(data=df_pyopus.iloc[[-1]], x = "area", y = "gain", ax=axes[0])
sns.scatterplot(data=df_pareto, x = "area", y = "bandwidth_3db_hz", ax=axes[1])
sns.scatterplot(data=df_pyopus.iloc[[-1]], x = "area", y = "f3db", ax=axes[1])
axes[0].set_xscale("log")
axes[1].set_xscale("log")

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

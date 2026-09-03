import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_shapeic = pd.read_csv("shapeic/results.csv")

df_shapeic["area"] = 2*df_shapeic["dp_width_m"]*df_shapeic["dp_length_m"]+2*df_shapeic["cm_width_m"]*df_shapeic["cm_length_m"]
df_shapeic = df_shapeic.sort_values(by="area")

print(df_shapeic.columns)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(data=df_shapeic, x = "area", y = "electrical_dc_gain_db", hue="cm_length_m", ax=axes[0])
sns.scatterplot(data=df_shapeic, x = "area", y = "electrical_bandwidth_3db_hz", hue="cm_length_m", ax=axes[1])

plt.grid()
plt.savefig("shapeic_exploration.png", dpi=300, bbox_inches="tight")


df_pyopus = pd.read_csv("pyopus/ota_opt_iterations.csv")

sns.scatterplot(data=df_pyopus.iloc[[-1]], x = "area", y = "gain")
##
##plt.xlabel("x")
##plt.ylabel("y")
##plt.grid()
#x = df["iteration"]
#y = df["gain"]
#for i in range(len(df) - 1):
#    plt.annotate(
#        "",
#        xy=(x.iloc[i + 1], y.iloc[i + 1]),
#        xytext=(x.iloc[i], y.iloc[i]),
#        arrowprops=dict(
#            arrowstyle="->",
#            lw=1.5,
#        ),
#    )
plt.savefig("sns_testplot.png", dpi=300, bbox_inches="tight")

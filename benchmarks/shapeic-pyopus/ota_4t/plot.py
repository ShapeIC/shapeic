import pandas as pd
import matplotlib.pyplot as plt

df_shapeic = pd.read_csv("shapeic/results.csv")

df_shapeic["area"] = 2*df_shapeic["dp_width_m"]*df_shapeic["dp_length_m"]+2*df_shapeic["cm_width_m"]*df_shapeic["cm_length_m"]
df_shapeic = df_shapeic.sort_values(by="area")
print(
    df_shapeic.assign(
        area=df_shapeic["area"],
        dp_width_m=df_shapeic["dp_width_m"] * 1e6,
        dp_length_m=df_shapeic["dp_length_m"] * 1e6,
        cm_width_m=df_shapeic["cm_width_m"] * 1e6,
        cm_length_m=df_shapeic["cm_length_m"] * 1e6,
    )[["area", "dp_width_m", "dp_length_m", "cm_width_m", "cm_length_m"]].head(20)
)

print(df_shapeic.columns)
plt.scatter(df_shapeic["area"], df_shapeic["electrical_dc_gain_db"])


df = pd.read_csv("pyopus/ota_opt_iterations.csv")

# Ver nombres de columnas
print(df.columns)

x = 2*df["w_n"]*df["l_n"]+2*df["w_p"]*df["l_p"]

# Graficar dos columnas
plt.scatter(x.iloc[-1], df["gain"].iloc[-1])
#
#plt.xlabel("x")
#plt.ylabel("y")
#plt.grid()
x = df["iteration"]
y = df["gain"]
for i in range(len(df) - 1):
    plt.annotate(
        "",
        xy=(x.iloc[i + 1], y.iloc[i + 1]),
        xytext=(x.iloc[i], y.iloc[i]),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.5,
        ),
    )

#plt.savefig("pyopus_iterations.png", dpi=300, bbox_inches="tight")


plt.xscale("log")
plt.savefig("shapeic_design_space.png", dpi=300, bbox_inches="tight")

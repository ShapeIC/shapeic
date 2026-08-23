from pathlib import Path
import subprocess
import csv

def render_spice_template(
    template_path: str | Path,
    output_path: str | Path,
    parameters: dict,
) -> Path:
    template_path = Path(template_path)
    output_path = Path(output_path)

    template = template_path.read_text()

    try:
        netlist = template.format(**parameters)
    except KeyError as e:
        raise ValueError(f"Missing parameter in template: {e.args[0]}") from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(netlist)

    return output_path


def run_ngspice(
    netlist_path: str | Path,
    log_path: str | Path | None = None,
):
    netlist_path = Path(netlist_path)

    if log_path is None:
        log_path = netlist_path.with_suffix(".log")
    else:
        log_path = Path(log_path)

    result = subprocess.run(
        [
            "ngspice",
            "-b",
            "-o",
            str(log_path),
            str(netlist_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ngspice failed with return code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return log_path

if __name__ == "__main__":
    print(f"{'w_dp':>6} {'l_dp':>6} {'ng_dp':>6} {'w_cm':>6} {'l_cm':>6} {'ng_cm':>6} {'Vout':>6} {'gain':>12} {'simgain':>12} {'gain_error':>10} {'bw_error':>10} {'gbw':>12} {'simgbw':>12} {'error':>6}")
    with open("../../shapeic-core/examples/ota_4t_v3/results.csv", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:

            w_dp = round(float(row["dp_width_m"])*1e6, 2)
            l_dp = round(float(row["dp_length_m"])*1e6, 2) 
            nf_dp = float(row["dp_nf"])
            w_cm = round(float(row["cm_width_m"])*1e6, 2)
            l_cm = round(float(row["cm_length_m"])*1e6, 2)
            nf_cm = float(row["cm_nf"])

            R1 = 100000000
            R2 = 0.9*R1/(float(row["vout_v"])-0.9)

            parameters = {"w_dp": w_dp, "l_dp": l_dp, "nf_dp": nf_dp,
                          "w_cm": w_cm, "l_cm": l_cm, "nf_cm": nf_cm,
                          "VDD": 1.5, "VREF": 0.9, "R2": R2,
                          "ac_results_path": "outputs/results.csv",
                          "corner_model_path": "~/SSTADEX-prev/IHP-Open-PDK/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib",
                          "psp_osdi": "~/SSTADEX-prev/IHP-Open-PDK/ihp-sg13g2/libs.tech/ngspice/osdi/psp103.osdi",
                          "psp_nqs_osdi": "~/SSTADEX-prev/IHP-Open-PDK/ihp-sg13g2/libs.tech/ngspice/osdi/psp103_nqs.osdi"}

            render_spice_template("ota_4t_validation.spice", "outputs/rendered.spice", parameters)
            run_ngspice("outputs/rendered.spice", "outputs/ngspice.log")
            
            with open("outputs/results.csv") as f:
                first_line = f.readline()

            results = [float(x) for x in first_line.split()]

            shapeic_gain = float(row["dc_gain_db"])
            simulation_gain = results[1]
            gain_error = (abs(simulation_gain-shapeic_gain)/simulation_gain)*100

            shapeic_bw = float(row["bandwidth_3db_hz"])
            simulation_bw = results[3]
            bw_error = (abs(simulation_bw-shapeic_bw)/simulation_bw)*100

            shapeic_gbw = float(row["unity_gain_hz"])
            simulation_gbw = results[5]
            gbw_error = (abs(shapeic_gbw - simulation_gbw)/simulation_gbw)*100

            shapeic_pm = float(row["phase_margin_deg"])
            simulation_pm = results[7]
            pm_error = (abs(simulation_pm-shapeic_pm)/simulation_pm)*100


            
            print(f"{w_dp:>6.2f} {l_dp:>6.1f} {nf_dp:>6.1f} {w_cm:>6.2f} {l_cm:>6.1f} {nf_cm:>6.1f} {float(row["vout_v"]):>6.2f} {shapeic_gain:>12.2f} {simulation_gain:>12.2f} {gain_error:>10.4f} {bw_error:>10.4f} {shapeic_gbw:>12.1f} {simulation_gbw:>12.1f} {gbw_error:>6.2f} {pm_error:>6.2f}")

from pathlib import Path

from pyopus.evaluator.performance import PerformanceEvaluator


PDK = Path(__file__).resolve().parents[4] / "pdks" / "ihp-sg13g2"
MODELS = PDK / "libs.tech" / "ngspice" / "models"


# ---------------------------------------------------------
# 1. Simulator
# ---------------------------------------------------------

heads = {
    "ngspice": {
        "simulator": "Ngspice",

        "moddefs": {
            # Esquina nominal de los MOS LV de SG13G2.
            "mos_tt": {
                "file": str(MODELS / "cornerMOSlv.lib"),
                "section": "mos_tt",
            },

            "ota": {
                "file": "ota.inc"
            },

            "tb": {
                "file": "tb.inc"
            },
        },
    }
}


# ---------------------------------------------------------
# 2. Analyses
# ---------------------------------------------------------

analyses = {

    "op": {
        "head": "ngspice",
        "modules": ["ota", "tb"],
        "saves": ["all()"],
        "command": "op()",
    },

    "ac": {
        "head": "ngspice",
        "modules": ["ota", "tb"],
        "saves": ["all()"],
        "command": "ac(1, 1e9, 'dec', 100)",
    },

    # Analisis sin simulacion para medidas que dependen solo de parametros.
    "blank": {
        "head": "ngspice",
        "command": None,
    },
}


# ---------------------------------------------------------
# 3. Corners
# ---------------------------------------------------------

corners = {
    "nominal": {
        "modules": ["mos_tt"],
        "params": {"temperature": 27},
    }
}


# ---------------------------------------------------------
# 4. Measures
# ---------------------------------------------------------

measures = {

    "gain": {
        "analysis": "ac",
        "corners": ["nominal"],

        # DC/low-frequency gain
        "expression":
            "20*np.log10(abs(v('out')[0]))",
    },

    "f3db": {
        "analysis": "ac",
        "corners": ["nominal"],

        # Primer cruce a -3 dB respecto de la ganancia AC maxima.
        "expression":
            "m.ACbandwidth(m.ACtf(v('out'), v('inp', 'inn')), scale())",
    },

    "isupply": {
        "analysis": "op",
        "corners": ["nominal"],

        "expression":
            "-i('vdd')",
    },

    "area": {
        "analysis": "blank",
        "corners": ["nominal"],

        # Dos NMOS y dos PMOS. No incluye contactos ni interconexion.
        "expression":
            "2*param['w_n']*param['l_n'] + 2*param['w_p']*param['l_p']",
    },
}


# ---------------------------------------------------------
# 5. Evaluator
# ---------------------------------------------------------

evaluator = PerformanceEvaluator(
    heads,
    analyses,
    measures,
    corners,
    debug=0,
)


# ---------------------------------------------------------
# Test one circuit
# ---------------------------------------------------------

params = {
    "w_n": 5e-6,
    "l_n": 0.5e-6,
    "w_p": 10e-6,
    "l_p": 0.5e-6,
}

if __name__ == "__main__":
    results, analysis_count = evaluator(params)

    print(results)
    print("SPICE analyses:", analysis_count)

    evaluator.finalize()

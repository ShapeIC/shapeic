from pyopus.evaluator.performance import PerformanceEvaluator


# ---------------------------------------------------------
# 1. Simulator
# ---------------------------------------------------------

heads = {
    "ngspice": {
        "simulator": "Ngspice",

        "moddefs": {
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
}


# ---------------------------------------------------------
# 3. Corners
# ---------------------------------------------------------

corners = {
    "nominal": {
        "modules": [],
        "params": {},
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

    "isupply": {
        "analysis": "op",
        "corners": ["nominal"],

        "expression":
            "-i('vdd')",
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
    "w_n": 10e-6,
    "l_n": 0.5e-6,
    "w_p": 20e-6,
    "l_p": 0.5e-6,
}

results, analysis_count = evaluator(params)

print(results)
print("SPICE analyses:", analysis_count)

evaluator.finalize()

"""Dimensionado nominal de la OTA IHP SG13G2 con PyOPUS."""

import numpy as np

from pyopus.evaluator.aggregate import Aggregator, Nabove, Nbelow
from pyopus.evaluator.performance import updateAnalysisCount
from pyopus.optimizer import optimizerClass

from ota import evaluator


# Orden de las variables en los vectores usados por el optimizador.
names = ["w_n", "l_n", "w_p", "l_p"]

# Limites electricos del modelo LV y punto inicial.
xlo = np.array([0.5e-6, 0.13e-6, 0.5e-6, 0.13e-6])
xhi = np.array([10e-6, 6.4e-6, 10e-6, 6.4e-6])
xinit = np.array([5e-6, 0.5e-6, 10e-6, 0.5e-6])

# Cada contribucion es positiva si se viola la especificacion y cero si se
# cumple. El optimizador busca un dimensionado que satisfaga todas.
requirements = [
    {
        "measure": "gain",
        "norm": Nabove(30.0, 3.0),       # ganancia >= 30 dB
    },
    {
        "measure": "f3db",
        "norm": Nabove(1e6, 100e3),    # frecuencia -3 dB >= 300 kHz
    },
    {
        "measure": "isupply",
        "norm": Nbelow(20e-6, 5e-6),     # corriente <= 20 uA
    },
    {
        "measure": "area",
        "norm": Nbelow(80e-12, 20e-12), # area ideal <= 80 um^2
    },
]


cost = Aggregator(evaluator, requirements, names, debug=0)
optimizer = optimizerClass("HookeJeeves")(
    cost,
    xlo=xlo,
    xhi=xhi,
    maxiter=300,
)
optimizer.reset(xinit)
optimizer.installPlugin(cost.getReporter())
optimizer.installPlugin(cost.getStopWhenAllSatisfied())

optimizer.run()

# Reevalua el mejor punto para dejar los resultados finales en el agregador.
final_cost = cost(optimizer.x)

print("\nResultado de la optimizacion")
print(f"Coste final: {final_cost:.6g}")
print(f"Iteracion del mejor punto: {optimizer.bestIter}")
print(cost.formatParameters())
print(cost.formatResults(nMeasureName=10, nCornerName=12))

analysis_count = {}
updateAnalysisCount(analysis_count, evaluator.analysisCount, optimizer.niter + 1)
print("Analisis SPICE aproximados:", analysis_count)

evaluator.finalize()

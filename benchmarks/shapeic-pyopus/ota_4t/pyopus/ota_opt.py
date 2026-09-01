"""Dimensionado nominal de la OTA IHP SG13G2 con PyOPUS."""

import numpy as np

from pyopus.evaluator.aggregate import Aggregator, Nabove, Nbelow
from pyopus.evaluator.performance import updateAnalysisCount
from pyopus.optimizer import optimizerClass

from ota import evaluator

def ota_opt():
    # Orden de las variables en los vectores usados por el optimizador.
    names = ["w_n", "l_n", "w_p", "l_p"]
    
    # Limites de busqueda y punto inicial. w_n y w_p son anchos totales; ng se
    # calcula automaticamente en ota.inc para mantener w/ng < 10 um.
    xlo = np.array([0.5e-6, 0.13e-6, 0.5e-6, 0.13e-6])
    xhi = np.array([100e-6, 6.4e-6, 100e-6, 6.4e-6])
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
            "norm": Nabove(1e6, 100e3),     # frecuencia -3 dB >= 1 MHz
        },
        {
            "measure": "isupply",
            "norm": Nbelow(20e-6, 5e-6),     # corriente <= 20 uA
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
    
    # ng es una consecuencia del ancho optimizado, no una variable de busqueda.
    ng_n = int(np.floor(optimizer.x[0] / 10e-6) + 1)
    ng_p = int(np.floor(optimizer.x[2] / 10e-6) + 1)
    print(f"ng_n: {ng_n}  (W/finger = {optimizer.x[0]/ng_n/1e-6:.6g} um)")
    print(f"ng_p: {ng_p}  (W/finger = {optimizer.x[2]/ng_p/1e-6:.6g} um)")
    print(cost.formatResults(nMeasureName=10, nCornerName=12))
    
    analysis_count = {}
    updateAnalysisCount(analysis_count, evaluator.analysisCount, optimizer.niter + 1)
    print("Analisis SPICE aproximados:", analysis_count)
    
    evaluator.finalize()

if __name__ == "__main__":
    ota_opt()


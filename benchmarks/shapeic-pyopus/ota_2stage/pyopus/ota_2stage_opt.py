"""Dimensionado nominal de la OTA IHP SG13G2 con PyOPUS."""

import csv
from pathlib import Path

import numpy as np

from pyopus.evaluator.aggregate import Aggregator, Nabove, Nbelow, Slinear2
from pyopus.evaluator.performance import updateAnalysisCount
from pyopus.optimizer import optimizerClass
from pyopus.optimizer.base import Plugin

from ota_2stage import evaluator


class CsvLogger(Plugin):
    def __init__(self, output, aggregator, names, measures):
        super().__init__()
        self.writer = csv.writer(output)
        self.aggregator = aggregator
        self.measures = measures
        self.writer.writerow(["iteration", "cost", *names, *measures])

    def __call__(self, x, cost, opt):
        results = {item["measure"]: item["worst"] for item in self.aggregator.results}
        self.writer.writerow(
            [opt.niter, cost, *x, *(results[measure] for measure in self.measures)]
        )


def ota_2stage_opt():
    # Orden de las variables en los vectores usados por el optimizador.
    names = ["w_n", "l_n", "w_p", "l_p", "w_c", "l_c", "r_comp", "c_comp",]
    
    # Limites de busqueda y punto inicial. w_n y w_p son anchos totales; ng se
    # calcula automaticamente en ota.inc para mantener w/ng < 10 um.
    xlo = np.array([0.5e-6, 0.13e-6, 0.5e-6, 0.13e-6, 0.5e-6, 0.13e-6, 1e2, 1e-15])
    xhi = np.array([100e-6, 6.4e-6, 100e-6, 6.4e-6, 1000e-6, 6.4e-6, 1e5, 1e-11])
    #xinit = np.array([44.12e-6, 6.4e-6, 56.88e-6, 6.4e-6, 31.25e-6, 0.8e-6, 1e4, 0.46e-12])
    xinit = np.array([5.0e-6, 0.5e-6, 5.0e-6, 0.5e-6, 5.0e-6, 0.5e-6, 1e4, 1e-12])

    # if I give it the start point as the better option from shapeic...
    #xinit = np.array([12.67e-6, 3.2e-6, 15e-6, 1.6e-6])
    
    # Cada contribucion es positiva si se viola la especificacion y cero si se
    # cumple. El optimizador busca un dimensionado que satisfaga todas.
    requirements = [
        {
            "measure": "gain",
            "norm": Nabove(70.0, 3.0),
            "shape": Slinear2(100.0, 0.0),
        },
        {
            "measure": "f3db",
            "norm": Nabove(1e4, 1e1),     # frecuencia -3 dB >= 1 MHz
            "shape": Slinear2(100.0, 0.0),
        },
        {
            "measure": "pm",
            "norm": Nabove(60.0, 5.0),
            "shape": Slinear2(100.0, 0.0),
        },
        {
            "measure": "area",
            "norm": Nbelow(1e-8, 20e-12), # area ideal <= 80 um^2
            "shape": Slinear2(1.0, 1.0),
        },
    ]
    
    
    cost = Aggregator(evaluator, requirements, names, debug=0)
    optimizer = optimizerClass("HookeJeeves")(
        cost,
        xlo=xlo,
        xhi=xhi,
        maxiter=1000,
    )
    optimizer.reset(xinit)
    optimizer.installPlugin(cost.getReporter())
    optimizer.installPlugin(cost.getStopWhenAllSatisfied())

    csv_path = Path(__file__).with_name("ota_opt_iterations.csv")
    with csv_path.open("w", newline="") as output:
        optimizer.installPlugin(CsvLogger(
            output,
            cost,
            names,
            [requirement["measure"] for requirement in requirements],
        ))
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
    ng_c = int(np.floor(optimizer.x[4] / 10e-6) + 1)
    print(f"ng_n: {ng_n}  (W/finger = {optimizer.x[0]/ng_n/1e-6:.6g} um)")
    print(f"ng_p: {ng_p}  (W/finger = {optimizer.x[2]/ng_p/1e-6:.6g} um)")
    print(f"ng_c: {ng_c}  (W/finger = {optimizer.x[4]/ng_c/1e-6:.6g} um)")
    print(cost.formatResults(nMeasureName=10, nCornerName=12))
    
    analysis_count = {}
    updateAnalysisCount(analysis_count, evaluator.analysisCount, optimizer.niter + 1)
    print("Analisis SPICE aproximados:", analysis_count)
    print("Iteraciones guardadas en:", csv_path)
    
    evaluator.finalize()

if __name__ == "__main__":
    ota_2stage_opt()

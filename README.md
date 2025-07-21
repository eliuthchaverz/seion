# Seion Core — Dinámica Seiónica 1D

Este repositorio contiene el motor computacional de simulación seiónica 1D basado en la Teoría Unificada de Conciencia y Campos Fractales (UTCF). Implementa la evolución de un campo complejo sobre un espacio no-asociativo con curvatura emergente, evaluando el error relativo entre curvatura y asociador.

## 🧠 ¿Qué es esto?

Una simulación numérica que implementa:

- Un producto ternario no-asociativo (`star`)
- Una descomposición tensorial CP-SVD del núcleo $K(x; y, z)$
- La evolución dinámica del campo $\phi$ con ecuaciones no-lineales complejas
- Monitoreo del error relativo de geometría: $\| R - A \|$

## 🚀 Archivos principales

| Archivo                          | Descripción                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `seion_dynamics_1d_final.py`     | Motor principal: evolución de campo $\phi$, cálculo de curvatura y error   |
| `scan_rel_err_final.py`          | Barrido de `alpha` y recolección de `rel.err` en CSV                       |
| `rel_err_vs_alpha_final.csv`     | (Salida esperada) Error relativo vs ruptura de simetría `alpha`           |

## 📦 Requisitos

Instala las dependencias:

```bash
pip install -r requirements.txt

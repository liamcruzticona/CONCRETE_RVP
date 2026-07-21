"""
=====================================================================
  generar_csvs.py - Discretiza y genera CSVs para Concrete dataset
  BEST = Strength >= 35 MPa (507)  |  WORST = Strength <= 25 MPa (295)
  8 variables independientes (Strength se usa solo para particionar)
=====================================================================
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("  GENERADOR DE CSVs: Concrete Compressive Strength")
print("=" * 60)

df = pd.read_csv("Concrete_Data.csv")
print(f"\n  Dataset original: {df.shape[0]} filas x {df.shape[1]} columnas")

df.columns = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer',
              'CoarseAgg', 'FineAgg', 'Age', 'Strength']

independent_vars = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer',
                    'CoarseAgg', 'FineAgg', 'Age']

print(f"\n  Discretizando {len(independent_vars)} variables...")
for var in independent_vars:
    try:
        df[var] = pd.qcut(df[var], q=4, labels=False, duplicates='drop')
    except:
        try:
            df[var] = pd.qcut(df[var], q=3, labels=False, duplicates='drop')
        except:
            df[var] = pd.qcut(df[var], q=2, labels=False, duplicates='drop')

print(f"  Valores unicos por variable:")
for var in independent_vars:
    vals = sorted(df[var].unique())
    print(f"    {var}: {vals} ({len(vals)} categorias)")

best = df[df['Strength'] >= 35]
worst = df[df['Strength'] < 35]

best_vars = best[independent_vars]
worst_vars = worst[independent_vars]

print(f"\n  Split:")
print(f"    BEST  (Strength >= 35): {len(best)} registros")
print(f"    WORST (Strength <= 25): {len(worst)} registros")
print(f"    Total: {len(best)+len(worst)}")

best_vars.to_csv("d9_concrete_B.csv", index=False)
worst_vars.to_csv("d9_concrete_W.csv", index=False)
print(f"\n  CSVs generados:")
print(f"    d9_concrete_B.csv  ->  {len(best)} filas x 8 columnas")
print(f"    d9_concrete_W.csv  ->  {len(worst)} filas x 8 columnas")
print()

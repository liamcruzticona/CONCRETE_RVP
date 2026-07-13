# CONCRETE - Analisis rVP con MAX (MI)

**Dataset:** Concrete Compressive Strength (UCI, 1030 registros)  
**Metodo:** Pipeline MAX (MI directa) + rVP (Rango de Variacion Permitido)

## Resultados rVP

| Variable | |Delta| | ¿Critica? | Interpretacion |
|----------|--------|:---:|-------------|
| **Age** | 2.17 | SI | Edad de curado cambia MASIVAMENTE entre BEST y WORST |
| **FlyAsh** | 1.51 | SI | Ceniza volante cambia su rol drasticamente |
| **FineAgg** | 1.13 | SI | Arena fina es discriminativa |
| **Slag** | 0.81 | SI | Escoria cambia su rol |
| **CoarseAgg** | 0.64 | SI | Grava gruesa es discriminativa |
| **Cement** | 0.51 | SI | Cemento tambien cambia |
| Superplasticizer | 0.03 | No | Aditivo es estable |
| Water | 0.02 | No | Agua es estable |

## Archivos

| Archivo | Funcion |
|---------|---------|
| `generar_csvs.py` | Discretizar + split BEST/WORST |
| `pipeline_rvp.py` | Pipeline MAX + rVP |
| `visual_concrete.py` | Visual interactiva paso a paso |
| `EXPLICACION_CODIGO.txt` | Explicacion detallada del codigo |

## Como ejecutar

```powershell
python generar_csvs.py      # Generar CSVs
python pipeline_rvp.py       # Pipeline + rVP
python visual_concrete.py    # Visual interactiva
```

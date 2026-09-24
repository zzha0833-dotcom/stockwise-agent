# Reproducible M5 benchmark

This benchmark was executed on the public M5 daily sales, calendar, and price data. Inventory, lead-time, and cost inputs remained simulated with fixed seeds and are not presented as real M5 inventory outcomes.

## Configuration

- Dataset: 100 items across `CA_1`, `TX_1`, and `WI_1`
- Observations: 582,300 daily rows and 300 store-item series
- History: 2011-01-29 through 2016-05-22
- Demand sparsity: 61.43% zero-demand observations
- Backtest: three rolling windows with a 28-day horizon
- Models: Seasonal Naive, Croston SBA, and Global LightGBM
- Repeated Agent runs: seeds 42, 43, and 44
- LLM mode: deterministic mock mode

Command:

```powershell
stockwise evaluate-m5 --items 100 --runs 3
```

## Measured results

| Metric | Mean result |
| --- | ---: |
| Agent task completion rate | 100% |
| Agent fallback rate | 0% |
| Selected model | Global LightGBM |
| Forecast WMAPE | 78.13% |
| Forecast MAE | 1.167 units/day |
| Forecast MASE | 0.957 |
| WMAPE improvement over Seasonal Naive | 21.56% |

The high absolute WMAPE reflects the strongly intermittent selected M5 series. MASE below 1.0 and the relative improvement are the more useful comparisons for this benchmark. Projected fill rate, order value, holding cost, stockout cost, and approval rate use simulated inventory scenarios and must not be described as observed M5 business outcomes.

The generated JSON and HTML reports are stored under `artifacts/evaluations/` and intentionally excluded from Git.

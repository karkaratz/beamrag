### Table A.1: Global Retrieval Ablation Summary (Pooled N=500)
**Sample**: Global (All Domains) ($N=500$, Winning Answers with $U_q$: $N=202$)

| Configuration | Jaccard Top-25 (%) | Top-10 Retention (%) | RBO (p=0.90) (%) | Drift Contamination (%) | Hub Intrusion (%) | Grounding Recall (%) | $\Delta$ Recall Loss (%) [95% CI] | Wilcoxon $p$-val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0: Full BeamRAG (Prod)** | 100.0% | 100.0% | 100.0% | 1.6% | 17.8% | 100.0% | 0.0% (Ref) | --- |
| C1: No Macro-Gating ($\tau_{\text{coh}}=0$) | 65.3% | 80.6% | 73.8% | 10.6% | 17.8% | 83.8% | +16.2% [12.0, 20.7] | $2.18e-10^*$ |
| C2: No Hub Penalty ($\deg>30$) | 46.0% | 74.5% | 52.0% | 1.6% | 34.2% | 73.6% | +26.4% [21.6, 31.2] | $2.60e-18^*$ |
| C3: No Relation Weights ($\alpha=1.0$) | 80.6% | 93.2% | 82.4% | 1.6% | 17.6% | 89.6% | +10.4% [7.1, 14.0] | $1.65e-09^*$ |
| C4: No Cross-Reranker ($\beta=1.0$) | 67.8% | 73.1% | 45.9% | 1.6% | 6.1% | 69.9% | +30.1% [25.0, 35.3] | $1.16e-17^*$ |
| C5: No Hop Decay ($\gamma=1.0$) | 84.6% | 98.9% | 85.2% | 1.6% | 18.5% | 94.6% | +5.4% [3.1, 8.1] | $1.26e-06^*$ |
| C6: No Leaf Dampening ($\deg\le1$) | 85.2% | 90.6% | 89.3% | 1.6% | 19.1% | 93.5% | +6.5% [3.7, 9.8] | $4.06e-05^*$ |
| *Baseline BFS (Reference Lower)* | 4.9% | 14.2% | 7.6% | 27.8% | 67.6% | 19.8% | +80.2% [75.7, 84.6] | $9.45e-36^*$ |


### Table A.2: Calibration-Holdout Suite Ablation Summary (N=375)
**Sample**: Holdout Suite (CS, Agri, Legal) ($N=375$, Winning Answers with $U_q$: $N=151$)

| Configuration | Jaccard Top-25 (%) | Top-10 Retention (%) | RBO (p=0.90) (%) | Drift Contamination (%) | Hub Intrusion (%) | Grounding Recall (%) | $\Delta$ Recall Loss (%) [95% CI] | Wilcoxon $p$-val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0: Full BeamRAG (Prod)** | 100.0% | 100.0% | 100.0% | 1.9% | 22.2% | 100.0% | 0.0% (Ref) | --- |
| C1: No Macro-Gating ($\tau_{\text{coh}}=0$) | 66.0% | 80.1% | 73.4% | 8.2% | 22.1% | 83.4% | +16.6% [11.6, 22.1] | $2.28e-08^*$ |
| C2: No Hub Penalty ($\deg>30$) | 40.4% | 70.0% | 48.5% | 1.9% | 43.0% | 70.7% | +29.3% [23.7, 35.0] | $1.18e-15^*$ |
| C3: No Relation Weights ($\alpha=1.0$) | 78.9% | 92.3% | 81.3% | 1.9% | 22.0% | 88.5% | +11.5% [7.3, 16.0] | $3.61e-08^*$ |
| C4: No Cross-Reranker ($\beta=1.0$) | 60.1% | 66.2% | 40.9% | 1.9% | 7.0% | 61.2% | +38.8% [32.6, 45.1] | $7.82e-17^*$ |
| C5: No Hop Decay ($\gamma=1.0$) | 82.9% | 98.8% | 85.7% | 1.9% | 23.0% | 93.4% | +6.6% [3.6, 10.1] | $6.01e-06^*$ |
| C6: No Leaf Dampening ($\deg\le1$) | 88.4% | 93.0% | 91.9% | 1.9% | 22.7% | 95.3% | +4.7% [2.0, 7.8] | 0.0011* |
| *Baseline BFS (Reference Lower)* | 5.8% | 16.4% | 8.2% | 22.9% | 68.5% | 23.8% | +76.2% [70.6, 81.4] | $4.08e-26^*$ |


### Table A.3: Domain Ablation — Computer Science (N=125)
**Sample**: Computer Science ($N=125$, Winning Answers with $U_q$: $N=47$)

| Configuration | Jaccard Top-25 (%) | Top-10 Retention (%) | RBO (p=0.90) (%) | Drift Contamination (%) | Hub Intrusion (%) | Grounding Recall (%) | $\Delta$ Recall Loss (%) [95% CI] | Wilcoxon $p$-val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0: Full BeamRAG (Prod)** | 100.0% | 100.0% | 100.0% | 5.6% | 5.0% | 100.0% | 0.0% (Ref) | --- |
| C1: No Macro-Gating ($\tau_{\text{coh}}=0$) | 59.1% | 76.4% | 68.6% | 17.7% | 3.0% | 88.8% | +11.2% [4.2, 19.6] | 0.0025* |
| C2: No Hub Penalty ($\deg>30$) | 59.1% | 88.4% | 62.3% | 5.6% | 10.6% | 85.2% | +14.8% [7.4, 23.1] | 0.0003* |
| C3: No Relation Weights ($\alpha=1.0$) | 83.4% | 95.4% | 82.8% | 5.6% | 5.0% | 89.9% | +10.1% [3.4, 18.0] | 0.0025* |
| C4: No Cross-Reranker ($\beta=1.0$) | 73.3% | 83.7% | 50.3% | 5.6% | 1.0% | 86.5% | +13.5% [7.2, 20.9] | 0.0003* |
| C5: No Hop Decay ($\gamma=1.0$) | 85.3% | 99.4% | 86.3% | 5.6% | 4.7% | 99.3% | +0.7% [0.0, 2.1] | 0.1587 |
| C6: No Leaf Dampening ($\deg\le1$) | 93.3% | 96.5% | 96.1% | 5.6% | 4.8% | 98.9% | +1.1% [0.0, 2.9] | 0.0899 |
| *Baseline BFS (Reference Lower)* | 8.8% | 24.8% | 13.3% | 37.7% | 51.0% | 37.7% | +62.3% [50.6, 73.7] | $2.06e-08^*$ |


### Table A.4: Domain Ablation — Agricultural Science (N=125)
**Sample**: Agricultural Science ($N=125$, Winning Answers with $U_q$: $N=54$)

| Configuration | Jaccard Top-25 (%) | Top-10 Retention (%) | RBO (p=0.90) (%) | Drift Contamination (%) | Hub Intrusion (%) | Grounding Recall (%) | $\Delta$ Recall Loss (%) [95% CI] | Wilcoxon $p$-val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0: Full BeamRAG (Prod)** | 100.0% | 100.0% | 100.0% | 0.0% | 18.0% | 100.0% | 0.0% (Ref) | --- |
| C1: No Macro-Gating ($\tau_{\text{coh}}=0$) | 53.2% | 70.6% | 60.8% | 6.0% | 19.5% | 72.5% | +27.6% [17.3, 38.6] | $3.70e-05^*$ |
| C2: No Hub Penalty ($\deg>30$) | 29.6% | 60.0% | 38.8% | 0.0% | 52.8% | 63.9% | +36.1% [27.2, 45.4] | $5.56e-08^*$ |
| C3: No Relation Weights ($\alpha=1.0$) | 78.5% | 93.7% | 83.8% | 0.0% | 16.8% | 89.4% | +10.6% [4.7, 17.8] | 0.0005* |
| C4: No Cross-Reranker ($\beta=1.0$) | 68.1% | 70.4% | 47.8% | 0.0% | 7.4% | 58.3% | +41.7% [31.1, 51.9] | $2.60e-07^*$ |
| C5: No Hop Decay ($\gamma=1.0$) | 81.0% | 99.2% | 84.3% | 0.0% | 19.6% | 90.3% | +9.7% [4.1, 16.7] | 0.0005* |
| C6: No Leaf Dampening ($\deg\le1$) | 82.7% | 88.7% | 86.6% | 0.0% | 20.5% | 95.9% | +4.1% [0.0, 9.6] | 0.0512 |
| *Baseline BFS (Reference Lower)* | 3.1% | 9.4% | 5.0% | 29.1% | 62.9% | 12.8% | +87.2% [80.1, 93.6] | $1.44e-11^*$ |


### Table A.5: Domain Ablation — Contractual Legal (N=125)
**Sample**: Contractual Legal ($N=125$, Winning Answers with $U_q$: $N=50$)

| Configuration | Jaccard Top-25 (%) | Top-10 Retention (%) | RBO (p=0.90) (%) | Drift Contamination (%) | Hub Intrusion (%) | Grounding Recall (%) | $\Delta$ Recall Loss (%) [95% CI] | Wilcoxon $p$-val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0: Full BeamRAG (Prod)** | 100.0% | 100.0% | 100.0% | 0.0% | 43.7% | 100.0% | 0.0% (Ref) | --- |
| C1: No Macro-Gating ($\tau_{\text{coh}}=0$) | 85.5% | 93.4% | 90.7% | 0.9% | 43.8% | 90.1% | +9.9% [4.0, 16.6] | 0.0038* |
| C2: No Hub Penalty ($\deg>30$) | 32.4% | 61.5% | 44.3% | 0.0% | 65.7% | 64.4% | +35.6% [25.4, 46.1] | $5.67e-07^*$ |
| C3: No Relation Weights ($\alpha=1.0$) | 74.9% | 87.8% | 77.5% | 0.0% | 44.2% | 86.2% | +13.8% [6.2, 22.6] | 0.0005* |
| C4: No Cross-Reranker ($\beta=1.0$) | 39.0% | 44.6% | 24.7% | 0.0% | 12.7% | 40.5% | +59.5% [49.5, 69.0] | $7.45e-09^*$ |
| C5: No Hop Decay ($\gamma=1.0$) | 82.5% | 97.9% | 86.4% | 0.0% | 44.8% | 91.2% | +8.8% [2.9, 16.1] | 0.0025* |
| C6: No Leaf Dampening ($\deg\le1$) | 89.3% | 93.8% | 93.0% | 0.0% | 43.0% | 91.3% | +8.7% [2.9, 16.0] | 0.0090* |
| *Baseline BFS (Reference Lower)* | 5.4% | 14.9% | 6.3% | 1.8% | 91.7% | 22.8% | +77.2% [68.6, 85.2] | $4.97e-10^*$ |


### Table A.6: Domain Ablation — Cross-Domain MIX (N=125)
**Sample**: Cross-Domain MIX ($N=125$, Winning Answers with $U_q$: $N=51$)

| Configuration | Jaccard Top-25 (%) | Top-10 Retention (%) | RBO (p=0.90) (%) | Drift Contamination (%) | Hub Intrusion (%) | Grounding Recall (%) | $\Delta$ Recall Loss (%) [95% CI] | Wilcoxon $p$-val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0: Full BeamRAG (Prod)** | 100.0% | 100.0% | 100.0% | 0.8% | 4.5% | 100.0% | 0.0% (Ref) | --- |
| C1: No Macro-Gating ($\tau_{\text{coh}}=0$) | 63.5% | 82.1% | 75.0% | 17.9% | 4.8% | 84.9% | +15.1% [7.4, 23.8] | 0.0010* |
| C2: No Hub Penalty ($\deg>30$) | 62.9% | 88.2% | 62.5% | 0.8% | 7.8% | 82.3% | +17.6% [9.1, 27.3] | 0.0002* |
| C3: No Relation Weights ($\alpha=1.0$) | 85.7% | 96.1% | 85.5% | 0.8% | 4.7% | 92.9% | +7.1% [2.5, 12.8] | 0.0056* |
| C4: No Cross-Reranker ($\beta=1.0$) | 90.8% | 93.8% | 60.9% | 0.8% | 3.2% | 95.7% | +4.3% [0.7, 9.4] | 0.0216* |
| C5: No Hop Decay ($\gamma=1.0$) | 89.6% | 99.0% | 83.7% | 0.8% | 5.2% | 98.1% | +1.9% [0.2, 4.3] | 0.0328* |
| C6: No Leaf Dampening ($\deg\le1$) | 75.7% | 83.5% | 81.4% | 0.8% | 8.4% | 88.0% | +12.0% [4.3, 20.8] | 0.0054* |
| *Baseline BFS (Reference Lower)* | 2.4% | 7.6% | 5.8% | 42.6% | 64.8% | 7.8% | +92.2% [85.5, 97.9] | $5.89e-12^*$ |

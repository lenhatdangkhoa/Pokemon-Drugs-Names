| Prompt                                              | Setting              | Confabulation Type       | gpt-5-chat       | GPT-4o-mini        |
| --------------------------------------------------- | -------------------- | ------------------------ | ---------------- | ------------------ |
| Drug Dosing Prompt (generic drug dataset)           | Default              | Overall confabulations   | 8.8% [5.9, 11.9] | 58.4% [53.9, 63.3] |
|                                                     |                      | Inherited confabulations | 3.6% [1.9, 5.5]  | 37.3% [32.4, 42.7] |
|                                                     |                      | Epistemic confabulations | 5.2% [2.9, 7.7]  | 21.1% [16.8, 24.9] |
|                                                     | Default + Mitigation | Overall confabulations   | 0.0% [0.0, 0.0]  | 0.7% [0.0, 1.7]    |
|                                                     |                      | Inherited confabulations | 0.0% [0.0, 0.0]  | 0.7% [0.0, 1.6]    |
|                                                     |                      | Epistemic confabulations | 0.0% [0.0, 0.0]  | 0.0% [0.0, 0.0]    |
|                                                     | Temp 0               | Overall confabulations   | 8.5% [5.7, 11.7] | 62.1% [57.1, 67.3] |
|                                                     |                      | Inherited confabulations | 3.6% [1.9, 5.7]  | 36.5% [30.8, 42.1] |
|                                                     |                      | Epistemic confabulations | 4.9% [2.5, 7.5]  | 25.6% [20.9, 30.3] |
| Drug Dosing Prompt (brand drug dataset)             | Default              | Overall confabulations   |                  | 57.6% [52.4, 63.2] |
|                                                     |                      | Inherited confabulations |                  | 55.5% [49.6, 61.1] |
|                                                     |                      | Epistemic confabulations |                  | 2.1% [0.8, 3.7]    |
|                                                     | Default + Mitigation | Overall confabulations   |                  | 1.1% [0.3, 2.1]    |
|                                                     |                      | Inherited confabulations |                  | 1.1% [0.3, 2.0]    |
|                                                     |                      | Epistemic confabulations |                  | 0.0% [0.0, 0.0]    |
|                                                     | Temp 0               | Overall confabulations   |                  | 60.5% [54.5, 66.3] |
|                                                     |                      | Inherited confabulations |                  | 58.5% [52.5, 64.0] |
|                                                     |                      | Epistemic confabulations |                  | 2.0% [0.5, 3.9]    |
| Medication Indication Prompt (generic drug dataset) | Default              | Overall confabulations   | 7.2% [4.7, 10.1] | 15.5% [12.0, 19.6] |
|                                                     |                      | Inherited confabulations | 4.1% [2.1, 6.4]  | 10.4% [7.2, 13.9]  |
|                                                     |                      | Epistemic confabulations | 3.1% [1.2, 5.2]  | 5.1% [3.1, 7.2]    |
|                                                     | Default + Mitigation | Overall confabulations   | 0.0% [0.0, 0.0]  | 0.0% [0.0, 0.0]    |
|                                                     |                      | Inherited confabulations | 0.0% [0.0, 0.0]  | 0.0% [0.0, 0.0]    |
|                                                     |                      | Epistemic confabulations | 0.0% [0.0, 0.0]  | 0.0% [0.0, 0.0]    |
|                                                     | Temp 0               | Overall confabulations   | 8.0% [5.1, 11.3] | 16.0% [12.3, 20.0] |
|                                                     |                      | Inherited confabulations | 4.9% [2.5, 7.9]  | 10.9% [7.3, 14.8]  |
|                                                     |                      | Epistemic confabulations | 3.1% [1.2, 5.2]  | 5.1% [3.1, 7.5]    |
| Medication Indication Prompt (brand drug dataset)   | Default              | Overall confabulations   |                  | 17.9% [13.7, 22.0] |
|                                                     |                      | Inherited confabulations |                  | 17.7% [13.1, 21.9] |
|                                                     |                      | Epistemic confabulations |                  | 0.1% [0.0, 0.4]    |
|                                                     | Default + Mitigation | Overall confabulations   |                  | 0.0% [0.0, 0.0]    |
|                                                     |                      | Inherited confabulations |                  | 0.0% [0.0, 0.0]    |
|                                                     |                      | Epistemic confabulations |                  | 0.0% [0.0, 0.0]    |
|                                                     | Temp 0               | Overall confabulations   |                  | 14.7% [10.7, 19.1] |
|                                                     |                      | Inherited confabulations |                  | 14.7% [10.4, 19.1] |
|                                                     |                      | Epistemic confabulations |                  | 0.0% [0.0, 0.0]    |

Rates reported as mean % (95% Confidence Interval). All the reported results were obtained from a bootstrapping sampling of size 1000 with 95% Confidence Interval (95% CI). Mean is calculated from case-level averages across 3 runs.

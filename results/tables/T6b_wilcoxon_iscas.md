| A                     | B                     |   median gap A-B (pp) |           p |      p Holm |
|:----------------------|:----------------------|----------------------:|------------:|------------:|
| Greedy (IG/cost)      | Greedy (balance/cost) |                 1.134 | 1.14088e-26 | 4.56353e-26 |
| Greedy (IG/cost)      | Rollout               |                 2.322 | 5.49848e-42 | 3.29909e-41 |
| Greedy (IG/cost)      | SA (static order)     |                -0.482 | 0.146922    | 0.146922    |
| Greedy (balance/cost) | Rollout               |                 0.571 | 5.30439e-12 | 1.06088e-11 |
| Greedy (balance/cost) | SA (static order)     |                -2.157 | 9.8872e-13  | 2.96616e-12 |
| Rollout               | SA (static order)     |                -3.157 | 2.04438e-33 | 1.02219e-32 |
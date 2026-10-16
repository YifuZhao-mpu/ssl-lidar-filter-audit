# T5 secondary analyses 3-4 (descriptive)

Kept counts and pseudo P/R against U ground truth (same-class BEV centre <= 2 m).

## Seed 0
- single_thr: kept 511,455  P 70.8  R 40.3
- perclass_only: kept 553,981  P 67.4  R 41.6
- best_view: kept 1,227,796  P 21.2  R 29.1
- consensus: kept 1,216,011  P 21.1  R 28.5
- perclass_and_consensus: kept 288,136  P 63.1  R 20.3
- C1: 35.5% of perclass_only kept mass out of vocabulary (196,424 / 553,981); 0.4% approximately mapped
- C2: pooled r 0.405; COCO-mapped r TP 0.38 / FP 0.51; vote lift TP 1.8 / FP 7.6
- C3 (consensus vs perclass_only): 6/8 in-vocabulary classes dominated, 2 trade; median keep ratio 0.52
  - bicycle: keep x0.34, dP +21.0, dR -8.8 [trade]
  - bus: keep x0.36, dP -14.8, dR -19.6 [dominated]
  - car: keep x0.88, dP -31.4, dR -20.7 [dominated]
  - construction_vehicle: keep x0.62, dP -35.9, dR -10.7 [dominated]
  - motorcycle: keep x0.38, dP +4.5, dR -14.7 [trade]
  - pedestrian: keep x0.30, dP -7.2, dR -28.1 [dominated]
  - trailer: keep x0.81, dP -27.7, dR -12.7 [dominated]
  - truck: keep x0.52, dP -12.7, dR -14.8 [dominated]
  - barrier: keep x3.37, dP -33.9, dR +14.2 [OOV]
  - traffic_cone: keep x7.34, dP -35.7, dR +20.2 [OOV]

## Seed 1
- single_thr: kept 524,528  P 69.3  R 40.5
- perclass_only: kept 571,194  P 65.5  R 41.7
- best_view: kept 1,245,293  P 21.2  R 29.4
- consensus: kept 1,232,879  P 21.0  R 28.8
- perclass_and_consensus: kept 306,198  P 60.4  R 20.6
- C1: 37.2% of perclass_only kept mass out of vocabulary (212,529 / 571,194); 0.4% approximately mapped
- C2: pooled r 0.448; COCO-mapped r TP 0.37 / FP 0.52; vote lift TP 1.8 / FP 7.6
- C3 (consensus vs perclass_only): 6/8 in-vocabulary classes dominated, 2 trade; median keep ratio 0.52
  - bicycle: keep x0.33, dP +25.6, dR -8.6 [trade]
  - bus: keep x0.37, dP -15.6, dR -20.7 [dominated]
  - car: keep x0.92, dP -32.9, dR -20.5 [dominated]
  - construction_vehicle: keep x0.66, dP -35.7, dR -10.3 [dominated]
  - motorcycle: keep x0.37, dP +4.2, dR -16.4 [trade]
  - pedestrian: keep x0.33, dP -7.4, dR -27.4 [dominated]
  - trailer: keep x0.87, dP -28.2, dR -11.1 [dominated]
  - truck: keep x0.52, dP -11.9, dR -14.6 [dominated]
  - barrier: keep x3.25, dP -30.6, dR +13.7 [OOV]
  - traffic_cone: keep x6.83, dP -33.5, dR +19.3 [OOV]

## Seed 2
- perclass_only: kept 567,169  P 66.0  R 41.7
- best_view: kept 1,292,891  P 20.6  R 29.6
- C1: 35.9% of perclass_only kept mass out of vocabulary (203,336 / 567,169); 0.4% approximately mapped
- C2: pooled r 0.419; COCO-mapped r TP 0.38 / FP 0.51; vote lift TP 1.8 / FP 7.5

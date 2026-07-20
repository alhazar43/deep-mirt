# A4 partial certification readout: NEURAL sub-matrix (banked, no slice pool)

Aggregated from the 24 completed production neural cells (2 profiles x 4 twins x 3 seeds).
Covers the ACTIVE posture (ACT P0/P1) and the four audit-battery gates on the neural
tracker (CG7 untrained-encoder null, CG8 drill contamination, CG9 order invariance,
CG10 direction audit). Does NOT cover PAS-G / MIX existence gates (those are the slice pool).

## kdd_matched
### syn_ng  (true mean rise/KC = 0.000)
- ACT_P0 pop_rise: 0.0064 [0.0063,0.0065]   ACT_P1 pop_rise: 0.0001 [0.0001,0.0001]
- CG7 untrained-null passed: 0/3   CG8 drill-contam passed: 0/3   CG9 order-inv passed: 0/3
- CG10 direction-violation fraction: 0.4540 [0.4206,0.4728] (0.5=chance; low=faithful)

### syn_kg  (true mean rise/KC = 0.673)
- ACT_P0 pop_rise: 0.0434 [0.0431,0.0435]   ACT_P1 pop_rise: 0.0575 [0.0565,0.0590]
- CG7 untrained-null passed: 0/3   CG8 drill-contam passed: 0/3   CG9 order-inv passed: 0/3
- CG10 direction-violation fraction: 0.3513 [0.2141,0.4229] (0.5=chance; low=faithful)

### syn_ns  (true mean rise/KC = 0.560)
- ACT_P0 pop_rise: 0.0602 [0.0595,0.0608]   ACT_P1 pop_rise: 0.0742 [0.0729,0.0768]
- CG7 untrained-null passed: 0/3   CG8 drill-contam passed: 0/3   CG9 order-inv passed: 0/3
- CG10 direction-violation fraction: 0.4338 [0.4208,0.4456] (0.5=chance; low=faithful)

### syn_sat  (true mean rise/KC = 0.673)
- ACT_P0 pop_rise: 0.0105 [0.0102,0.0108]   ACT_P1 pop_rise: 0.0227 [0.0141,0.0282]
- CG7 untrained-null passed: 0/3   CG8 drill-contam passed: 0/3   CG9 order-inv passed: 0/3
- CG10 direction-violation fraction: 0.2191 [0.0283,0.4260] (0.5=chance; low=faithful)

## ednet_matched
### syn_ng  (true mean rise/KC = 0.000)
- ACT_P0 pop_rise: 0.0089 [0.0078,0.0099]   ACT_P1 pop_rise: 0.0277 [0.0268,0.0294]
- CG7 untrained-null passed: 0/3   CG8 drill-contam passed: 0/3   CG9 order-inv passed: 0/3
- CG10 direction-violation fraction: 0.4062 [0.3902,0.4205] (0.5=chance; low=faithful)

### syn_kg  (true mean rise/KC = 0.869)
- ACT_P0 pop_rise: 0.1268 [0.1267,0.1270]   ACT_P1 pop_rise: 0.1599 [0.1584,0.1609]
- CG7 untrained-null passed: 0/3   CG8 drill-contam passed: 0/3   CG9 order-inv passed: 1/3
- CG10 direction-violation fraction: 0.3833 [0.3630,0.4041] (0.5=chance; low=faithful)

### syn_ns  (true mean rise/KC = 0.655)
- ACT_P0 pop_rise: 0.1234 [0.1230,0.1238]   ACT_P1 pop_rise: 0.1428 [0.1415,0.1435]
- CG7 untrained-null passed: 0/3   CG8 drill-contam passed: 0/3   CG9 order-inv passed: 2/3
- CG10 direction-violation fraction: 0.3961 [0.3842,0.4075] (0.5=chance; low=faithful)

### syn_sat  (true mean rise/KC = 0.869)
- ACT_P0 pop_rise: 0.0326 [0.0320,0.0335]   ACT_P1 pop_rise: 0.0431 [0.0427,0.0435]
- CG7 untrained-null passed: 0/3   CG8 drill-contam passed: 0/3   CG9 order-inv passed: 0/3
- CG10 direction-violation fraction: 0.3954 [0.3742,0.4234] (0.5=chance; low=faithful)

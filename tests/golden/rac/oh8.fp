IRREPS 10
  IRREP name=0+ kind=GROUND mult=4 dim=1
  IRREP name=^0+ kind=GROUND mult=1 dim=1
  IRREP name=2+ kind=GROUND mult=5 dim=2
  IRREP name=1+ kind=GROUND mult=4 dim=3
  IRREP name=^1+ kind=GROUND mult=6 dim=3
  IRREP name=0- kind=EXCITE mult=2 dim=1
  IRREP name=^0- kind=EXCITE mult=3 dim=1
  IRREP name=2- kind=EXCITE mult=5 dim=2
  IRREP name=1- kind=EXCITE mult=7 dim=3
  IRREP name=^1- kind=EXCITE mult=8 dim=3
BLOCKS 32
  BLOCK 0 kind=TRANSI bra=0+ op=1- ket=1- geometry=MULTIPOLE n_bra=4 n_ket=7 adds=3
    ADD 0 matrix_idx=1 bra=1 ket=1 nbra=2 nket=3 coeff=0.99999999999999989
    ADD 1 matrix_idx=11 bra=3 ket=4 nbra=2 nket=3 coeff=-0.33333333333333287
    ADD 2 matrix_idx=12 bra=3 ket=7 nbra=2 nket=1 coeff=-0.33333333333333309
  BLOCK 1 kind=TRANSI bra=^0+ op=1- ket=^1- geometry=MULTIPOLE n_bra=1 n_ket=8 adds=3
    ADD 0 matrix_idx=8 bra=1 ket=1 nbra=1 nket=4 coeff=-0.37796447300922698
    ADD 1 matrix_idx=9 bra=1 ket=5 nbra=1 nket=3 coeff=0.37796447300922686
    ADD 2 matrix_idx=10 bra=1 ket=8 nbra=1 nket=1 coeff=0.37796447300922698
  BLOCK 2 kind=TRANSI bra=2+ op=1- ket=1- geometry=MULTIPOLE n_bra=5 n_ket=7 adds=4
    ADD 0 matrix_idx=5 bra=1 ket=1 nbra=3 nket=3 coeff=0.6324555320336761
    ADD 1 matrix_idx=7 bra=1 ket=4 nbra=3 nket=3 coeff=-0.50709255283710997
    ADD 2 matrix_idx=11 bra=4 ket=4 nbra=2 nket=3 coeff=-0.28171808490950528
    ADD 3 matrix_idx=12 bra=4 ket=7 nbra=2 nket=1 coeff=0.39440531887330749
  BLOCK 3 kind=TRANSI bra=2+ op=1- ket=^1- geometry=MULTIPOLE n_bra=5 n_ket=8 adds=4
    ADD 0 matrix_idx=6 bra=1 ket=1 nbra=3 nket=4 coeff=-0.63245553203367555
    ADD 1 matrix_idx=7 bra=1 ket=5 nbra=3 nket=3 coeff=-0.3779644730092272
    ADD 2 matrix_idx=11 bra=4 ket=5 nbra=2 nket=3 coeff=0.37796447300922686
    ADD 3 matrix_idx=12 bra=4 ket=8 nbra=2 nket=1 coeff=-0.25819888974716082
  BLOCK 4 kind=TRANSI bra=1+ op=1- ket=0- geometry=MULTIPOLE n_bra=4 n_ket=2 adds=3
    ADD 0 matrix_idx=2 bra=1 ket=1 nbra=1 nket=1 coeff=0.99999999999999978
    ADD 1 matrix_idx=10 bra=2 ket=2 nbra=1 nket=1 coeff=-0.33333333333333298
    ADD 2 matrix_idx=12 bra=3 ket=2 nbra=2 nket=1 coeff=0.33333333333333309
  BLOCK 5 kind=TRANSI bra=1+ op=1- ket=2- geometry=MULTIPOLE n_bra=4 n_ket=5 adds=4
    ADD 0 matrix_idx=4 bra=1 ket=1 nbra=1 nket=4 coeff=0.63245553203367588
    ADD 1 matrix_idx=8 bra=2 ket=1 nbra=1 nket=4 coeff=-0.50709255283710974
    ADD 2 matrix_idx=10 bra=2 ket=5 nbra=1 nket=1 coeff=-0.28171808490950528
    ADD 3 matrix_idx=12 bra=3 ket=5 nbra=2 nket=1 coeff=-0.39440531887330749
  BLOCK 6 kind=TRANSI bra=1+ op=1- ket=1- geometry=MULTIPOLE n_bra=4 n_ket=7 adds=5
    ADD 0 matrix_idx=3 bra=1 ket=1 nbra=1 nket=3 coeff=0.99999999999999989
    ADD 1 matrix_idx=9 bra=2 ket=4 nbra=1 nket=3 coeff=-0.40089186286863626
    ADD 2 matrix_idx=10 bra=2 ket=7 nbra=1 nket=1 coeff=-0.45643546458763806
    ADD 3 matrix_idx=11 bra=3 ket=4 nbra=2 nket=3 coeff=0.45643546458763806
    ADD 4 matrix_idx=12 bra=3 ket=7 nbra=2 nket=1 coeff=0.091287092917527693
  BLOCK 7 kind=TRANSI bra=1+ op=1- ket=^1- geometry=MULTIPOLE n_bra=4 n_ket=8 adds=6
    ADD 0 matrix_idx=4 bra=1 ket=1 nbra=1 nket=4 coeff=0.77459666924148318
    ADD 1 matrix_idx=8 bra=2 ket=1 nbra=1 nket=4 coeff=0.41403933560541228
    ADD 2 matrix_idx=9 bra=2 ket=5 nbra=1 nket=3 coeff=0.51754916950676533
    ADD 3 matrix_idx=10 bra=2 ket=8 nbra=1 nket=1 coeff=-0.1725163898355884
    ADD 4 matrix_idx=11 bra=3 ket=5 nbra=2 nket=3 coeff=0.35355339059327345
    ADD 5 matrix_idx=12 bra=3 ket=8 nbra=2 nket=1 coeff=-0.24152294576982375
  BLOCK 8 kind=TRANSI bra=^1+ op=1- ket=^0- geometry=MULTIPOLE n_bra=6 n_ket=3 adds=3
    ADD 0 matrix_idx=7 bra=1 ket=1 nbra=3 nket=3 coeff=-0.37796447300922698
    ADD 1 matrix_idx=9 bra=4 ket=1 nbra=1 nket=3 coeff=-0.37796447300922686
    ADD 2 matrix_idx=11 bra=5 ket=1 nbra=2 nket=3 coeff=0.37796447300922703
  BLOCK 9 kind=TRANSI bra=^1+ op=1- ket=2- geometry=MULTIPOLE n_bra=6 n_ket=5 adds=4
    ADD 0 matrix_idx=6 bra=1 ket=1 nbra=3 nket=4 coeff=0.63245553203367544
    ADD 1 matrix_idx=8 bra=4 ket=1 nbra=1 nket=4 coeff=-0.37796447300922709
    ADD 2 matrix_idx=10 bra=4 ket=5 nbra=1 nket=1 coeff=0.37796447300922681
    ADD 3 matrix_idx=12 bra=5 ket=5 nbra=2 nket=1 coeff=0.25819888974716093
  BLOCK 10 kind=TRANSI bra=^1+ op=1- ket=1- geometry=MULTIPOLE n_bra=6 n_ket=7 adds=6
    ADD 0 matrix_idx=5 bra=1 ket=1 nbra=3 nket=3 coeff=0.77459666924148329
    ADD 1 matrix_idx=7 bra=1 ket=4 nbra=3 nket=3 coeff=0.41403933560541228
    ADD 2 matrix_idx=9 bra=4 ket=4 nbra=1 nket=3 coeff=-0.51754916950676522
    ADD 3 matrix_idx=10 bra=4 ket=7 nbra=1 nket=1 coeff=0.35355339059327345
    ADD 4 matrix_idx=11 bra=5 ket=4 nbra=2 nket=3 coeff=-0.17251638983558837
    ADD 5 matrix_idx=12 bra=5 ket=7 nbra=2 nket=1 coeff=0.24152294576982378
  BLOCK 11 kind=TRANSI bra=^1+ op=1- ket=^1- geometry=MULTIPOLE n_bra=6 n_ket=8 adds=7
    ADD 0 matrix_idx=6 bra=1 ket=1 nbra=3 nket=4 coeff=0.44721359549995748
    ADD 1 matrix_idx=7 bra=1 ket=5 nbra=3 nket=3 coeff=-0.53452248382484846
    ADD 2 matrix_idx=8 bra=4 ket=1 nbra=1 nket=4 coeff=0.53452248382484835
    ADD 3 matrix_idx=9 bra=4 ket=5 nbra=1 nket=3 coeff=-0.13363062095621223
    ADD 4 matrix_idx=10 bra=4 ket=8 nbra=1 nket=1 coeff=0.40089186286863637
    ADD 5 matrix_idx=11 bra=5 ket=5 nbra=2 nket=3 coeff=-0.40089186286863643
    ADD 6 matrix_idx=12 bra=5 ket=8 nbra=2 nket=1 coeff=-0.4564354645876379
  BLOCK 12 kind=GROUND bra=0+ op=0+ ket=0+ geometry=HAMILTONIAN n_bra=4 n_ket=4 adds=2
    ADD 0 matrix_idx=13 bra=1 ket=1 nbra=2 nket=2 coeff=1
    ADD 1 matrix_idx=17 bra=3 ket=3 nbra=2 nket=2 coeff=0.33333333333333331
  BLOCK 13 kind=GROUND bra=0+ op=0+ ket=0+ geometry=10DQ n_bra=4 n_ket=4 adds=3
    ADD 0 matrix_idx=18 bra=1 ket=3 nbra=2 nket=2 coeff=1.0954451150103335
    ADD 1 matrix_idx=28 bra=3 ket=1 nbra=2 nket=2 coeff=1.0954451150103335
    ADD 2 matrix_idx=32 bra=3 ket=3 nbra=2 nket=2 coeff=0.52357031440473589
  BLOCK 14 kind=GROUND bra=^0+ op=0+ ket=^0+ geometry=HAMILTONIAN n_bra=1 n_ket=1 adds=1
    ADD 0 matrix_idx=16 bra=1 ket=1 nbra=1 nket=1 coeff=0.3779644730092272
  BLOCK 15 kind=GROUND bra=^0+ op=0+ ket=^0+ geometry=10DQ n_bra=1 n_ket=1 adds=1
    ADD 0 matrix_idx=26 bra=1 ket=1 nbra=1 nket=1 coeff=-0.80903983495589016
  BLOCK 16 kind=GROUND bra=2+ op=0+ ket=2+ geometry=HAMILTONIAN n_bra=5 n_ket=5 adds=2
    ADD 0 matrix_idx=15 bra=1 ket=1 nbra=3 nket=3 coeff=0.63245553203367588
    ADD 1 matrix_idx=17 bra=4 ket=4 nbra=2 nket=2 coeff=0.47140452079103168
  BLOCK 17 kind=GROUND bra=2+ op=0+ ket=2+ geometry=10DQ n_bra=5 n_ket=5 adds=4
    ADD 0 matrix_idx=21 bra=1 ket=1 nbra=3 nket=3 coeff=0.84852813742385702
    ADD 1 matrix_idx=23 bra=1 ket=4 nbra=3 nket=2 coeff=0.93419873299382816
    ADD 2 matrix_idx=30 bra=4 ket=1 nbra=2 nket=3 coeff=0.93419873299382816
    ADD 3 matrix_idx=32 bra=4 ket=4 nbra=2 nket=2 coeff=0.10577717706958907
  BLOCK 18 kind=GROUND bra=1+ op=0+ ket=1+ geometry=HAMILTONIAN n_bra=4 n_ket=4 adds=3
    ADD 0 matrix_idx=14 bra=1 ket=1 nbra=1 nket=1 coeff=1
    ADD 1 matrix_idx=16 bra=2 ket=2 nbra=1 nket=1 coeff=0.65465367070797709
    ADD 2 matrix_idx=17 bra=3 ket=3 nbra=2 nket=2 coeff=0.57735026918962573
  BLOCK 19 kind=GROUND bra=1+ op=0+ ket=1+ geometry=10DQ n_bra=4 n_ket=4 adds=8
    ADD 0 matrix_idx=19 bra=1 ket=2 nbra=1 nket=1 coeff=-1.0954451150103324
    ADD 1 matrix_idx=20 bra=1 ket=3 nbra=1 nket=2 coeff=1.095445115010333
    ADD 2 matrix_idx=24 bra=2 ket=1 nbra=1 nket=1 coeff=-1.0954451150103324
    ADD 3 matrix_idx=26 bra=2 ket=2 nbra=1 nket=1 coeff=0.70064904974537145
    ADD 4 matrix_idx=27 bra=2 ket=3 nbra=1 nket=2 coeff=0.61791438065332516
    ADD 5 matrix_idx=29 bra=3 ket=1 nbra=2 nket=1 coeff=-1.095445115010333
    ADD 6 matrix_idx=31 bra=3 ket=2 nbra=2 nket=1 coeff=-0.61791438065332516
    ADD 7 matrix_idx=32 bra=3 ket=3 nbra=2 nket=2 coeff=0.45342519294190725
  BLOCK 20 kind=GROUND bra=^1+ op=0+ ket=^1+ geometry=HAMILTONIAN n_bra=6 n_ket=6 adds=3
    ADD 0 matrix_idx=15 bra=1 ket=1 nbra=3 nket=3 coeff=0.7745966692414834
    ADD 1 matrix_idx=16 bra=4 ket=4 nbra=1 nket=1 coeff=0.65465367070797709
    ADD 2 matrix_idx=17 bra=5 ket=5 nbra=2 nket=2 coeff=0.57735026918962573
  BLOCK 21 kind=GROUND bra=^1+ op=0+ ket=^1+ geometry=10DQ n_bra=6 n_ket=6 adds=9
    ADD 0 matrix_idx=21 bra=1 ket=1 nbra=3 nket=3 coeff=-0.69282032302755026
    ADD 1 matrix_idx=22 bra=1 ket=4 nbra=3 nket=1 coeff=1.095445115010331
    ADD 2 matrix_idx=23 bra=1 ket=5 nbra=3 nket=2 coeff=0.57207755354735512
    ADD 3 matrix_idx=25 bra=4 ket=1 nbra=1 nket=3 coeff=-1.095445115010331
    ADD 4 matrix_idx=26 bra=4 ket=4 nbra=1 nket=1 coeff=-0.23354968324845732
    ADD 5 matrix_idx=27 bra=4 ket=5 nbra=1 nket=2 coeff=-0.90453403373328967
    ADD 6 matrix_idx=30 bra=5 ket=1 nbra=2 nket=3 coeff=0.57207755354735512
    ADD 7 matrix_idx=31 bra=5 ket=4 nbra=2 nket=1 coeff=0.90453403373328967
    ADD 8 matrix_idx=32 bra=5 ket=5 nbra=2 nket=2 coeff=-0.84207535832068448
  BLOCK 22 kind=GROUND bra=0- op=0+ ket=0- geometry=HAMILTONIAN n_bra=2 n_ket=2 adds=2
    ADD 0 matrix_idx=33 bra=1 ket=1 nbra=1 nket=1 coeff=1
    ADD 1 matrix_idx=37 bra=2 ket=2 nbra=1 nket=1 coeff=0.33333333333333331
  BLOCK 23 kind=GROUND bra=0- op=0+ ket=0- geometry=10DQ n_bra=2 n_ket=2 adds=3
    ADD 0 matrix_idx=38 bra=1 ket=2 nbra=1 nket=1 coeff=1.0954451150103335
    ADD 1 matrix_idx=48 bra=2 ket=1 nbra=1 nket=1 coeff=1.0954451150103335
    ADD 2 matrix_idx=52 bra=2 ket=2 nbra=1 nket=1 coeff=0.52357031440473589
  BLOCK 24 kind=GROUND bra=^0- op=0+ ket=^0- geometry=HAMILTONIAN n_bra=3 n_ket=3 adds=1
    ADD 0 matrix_idx=36 bra=1 ket=1 nbra=3 nket=3 coeff=0.3779644730092272
  BLOCK 25 kind=GROUND bra=^0- op=0+ ket=^0- geometry=10DQ n_bra=3 n_ket=3 adds=1
    ADD 0 matrix_idx=46 bra=1 ket=1 nbra=3 nket=3 coeff=-0.80903983495589016
  BLOCK 26 kind=GROUND bra=2- op=0+ ket=2- geometry=HAMILTONIAN n_bra=5 n_ket=5 adds=2
    ADD 0 matrix_idx=35 bra=1 ket=1 nbra=4 nket=4 coeff=0.63245553203367588
    ADD 1 matrix_idx=37 bra=5 ket=5 nbra=1 nket=1 coeff=0.47140452079103168
  BLOCK 27 kind=GROUND bra=2- op=0+ ket=2- geometry=10DQ n_bra=5 n_ket=5 adds=4
    ADD 0 matrix_idx=41 bra=1 ket=1 nbra=4 nket=4 coeff=0.84852813742385702
    ADD 1 matrix_idx=43 bra=1 ket=5 nbra=4 nket=1 coeff=0.93419873299382816
    ADD 2 matrix_idx=50 bra=5 ket=1 nbra=1 nket=4 coeff=0.93419873299382816
    ADD 3 matrix_idx=52 bra=5 ket=5 nbra=1 nket=1 coeff=0.10577717706958907
  BLOCK 28 kind=GROUND bra=1- op=0+ ket=1- geometry=HAMILTONIAN n_bra=7 n_ket=7 adds=3
    ADD 0 matrix_idx=34 bra=1 ket=1 nbra=3 nket=3 coeff=1
    ADD 1 matrix_idx=36 bra=4 ket=4 nbra=3 nket=3 coeff=0.65465367070797709
    ADD 2 matrix_idx=37 bra=7 ket=7 nbra=1 nket=1 coeff=0.57735026918962573
  BLOCK 29 kind=GROUND bra=1- op=0+ ket=1- geometry=10DQ n_bra=7 n_ket=7 adds=8
    ADD 0 matrix_idx=39 bra=1 ket=4 nbra=3 nket=3 coeff=-1.0954451150103324
    ADD 1 matrix_idx=40 bra=1 ket=7 nbra=3 nket=1 coeff=1.095445115010333
    ADD 2 matrix_idx=44 bra=4 ket=1 nbra=3 nket=3 coeff=-1.0954451150103324
    ADD 3 matrix_idx=46 bra=4 ket=4 nbra=3 nket=3 coeff=0.70064904974537145
    ADD 4 matrix_idx=47 bra=4 ket=7 nbra=3 nket=1 coeff=0.61791438065332516
    ADD 5 matrix_idx=49 bra=7 ket=1 nbra=1 nket=3 coeff=-1.095445115010333
    ADD 6 matrix_idx=51 bra=7 ket=4 nbra=1 nket=3 coeff=-0.61791438065332516
    ADD 7 matrix_idx=52 bra=7 ket=7 nbra=1 nket=1 coeff=0.45342519294190725
  BLOCK 30 kind=GROUND bra=^1- op=0+ ket=^1- geometry=HAMILTONIAN n_bra=8 n_ket=8 adds=3
    ADD 0 matrix_idx=35 bra=1 ket=1 nbra=4 nket=4 coeff=0.7745966692414834
    ADD 1 matrix_idx=36 bra=5 ket=5 nbra=3 nket=3 coeff=0.65465367070797709
    ADD 2 matrix_idx=37 bra=8 ket=8 nbra=1 nket=1 coeff=0.57735026918962573
  BLOCK 31 kind=GROUND bra=^1- op=0+ ket=^1- geometry=10DQ n_bra=8 n_ket=8 adds=9
    ADD 0 matrix_idx=41 bra=1 ket=1 nbra=4 nket=4 coeff=-0.69282032302755026
    ADD 1 matrix_idx=42 bra=1 ket=5 nbra=4 nket=3 coeff=1.095445115010331
    ADD 2 matrix_idx=43 bra=1 ket=8 nbra=4 nket=1 coeff=0.57207755354735512
    ADD 3 matrix_idx=45 bra=5 ket=1 nbra=3 nket=4 coeff=-1.095445115010331
    ADD 4 matrix_idx=46 bra=5 ket=5 nbra=3 nket=3 coeff=-0.23354968324845732
    ADD 5 matrix_idx=47 bra=5 ket=8 nbra=3 nket=1 coeff=-0.90453403373328967
    ADD 6 matrix_idx=50 bra=8 ket=1 nbra=1 nket=4 coeff=0.57207755354735512
    ADD 7 matrix_idx=51 bra=8 ket=5 nbra=1 nket=3 coeff=0.90453403373328967
    ADD 8 matrix_idx=52 bra=8 ket=8 nbra=1 nket=1 coeff=-0.84207535832068448

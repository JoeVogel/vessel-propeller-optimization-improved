import numpy as np

from vessel_propeller_evaluation.still_watter_resistance import calculate_still_watter_resistance
from vessel_propeller_evaluation.still_watter_propulsion_factors import calculate_propulsion_factors
from vessel_propeller_evaluation.propeller_performance import PropellerPerformance

def evaluate(V_S, D, Z, AEdAO, PdD):
    # Computation of total resistance in still water
    L = 22
    B = 9.6
    T_F = 1.089
    T_A = 1.089
    nabla = 204.1
    LCB = 0
    A_BT = 0
    h_B = 0
    C_M = 0.9761
    C_WP = 0.99999
    A_T = 0
    Cstern = 10
    apk2 = 0
    S_APP = 0

    Rtotal, C_B, C_P, S, apk1, C_F, C_A, Fn, i_E = calculate_still_watter_resistance(L, B, T_F, T_A, nabla, LCB, A_BT, h_B, C_M, C_WP, A_T, S_APP, apk2, Cstern, V_S)

    # print('SWResist')
    # print(f"{Rtotal=}")
    # print(f"{C_B=}")
    # print(f"{C_P=}")
    # print(f"{S=}")
    # print(f"{apk1=}")
    # print(f"{C_F=}")
    # print(f"{C_A=}")
    # print()
    # print()

    # Computation of propulsion factors in still water
    zP = 2
    
    t, w, etaR = calculate_propulsion_factors(L, B, C_B, C_P, C_M, T_F, T_A, LCB, S_APP, S, apk1, apk2, C_F, C_A, Cstern, D, AEdAO, PdD, zP)
     
    # print('PropFac')
    # print(f"{t=}")
    # print(f"{w=}")
    # print(f"{etaR=}")
    # print()
    # print()
          
    hk = 0.5
    try:
        evaluator = PropellerPerformance()
        # Computation of propeller performance in still water
        n, P_O, etaO, t075dD, tmin075dD, tal07R, cavLim, Vtip, Vtipmax = evaluator.calculate_propeller_performance(Rtotal, V_S, t, w, etaR, zP, Z, D, PdD, AEdAO, hk, T_A)

        # print('Constraints')
        # print(f"{n=}")
        # print(f"{P_O=}")
        # print(f"{etaO=}")
        # print(f"{t075dD=}")
        # print(f"{tmin075dD=}")
        # print(f"{tal07R=}")
        # print(f"{cavLim=}")
        # print(f"{Vtip=}")
        # print(f"{Vtipmax=}")
        # print()
        # print()

        # Computation of brake power in still water
        etaS = 0.99 # shaft efficience []
        etaGB = 1   # gearbox efficience []
        
        P_B = P_O / (etaR * etaS * etaGB)
    except Exception as e:
        print(f"Error: {e}")
        P_B = 0
        n = 0
        etaO = 0
        etaR = 0

    # return P_B, n, etaO, etaR, t075dD, tmin075dD, tal07R, cavLim, Vtip, Vtipmax
    return P_B, t075dD, tmin075dD, tal07R, cavLim, Vtip, Vtipmax
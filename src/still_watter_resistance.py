import numpy as np

def calculate_still_watter_resistance(L, B, T_F, T_A, nabla, LCB, A_BT, h_B, C_M, C_WP, A_T, S_APP, apk2, Cstern, V_S):
    # Constantes
    g = 9.80665          # aceleração de gravidade [m/s**2]
    nu = 1.1390e-6       # viscosidade cinemática da água doce a 15°C [m**2/s]
    rho = 999            # densidade da água doce [kg/m**3]
    d = -0.9
    
    V_S = V_S * 1852 / 3600  # velocidade de serviço [m/s]
    Fn = V_S / (g * L) ** 0.5  # número de Froude
    Rn = V_S * L / nu  # número de Reynolds
    T_M = (T_A + T_F) / 2  # calado médio [m]
    C_P = nabla / (C_M * L * B * T_M)  # coeficiente prismático
    C_B = nabla / (L * B * T_M)  # coeficiente de bloco

    # Cálculos preliminares para a previsão da resistência
    L_R = (1 - C_P + 0.06 * C_P * LCB / (4 * C_P - 1)) * L

    c14 = 1 + 0.011 * Cstern

    BdL = B / L
    L3dnabla = L**3 / nabla
    TdL = T_M / L

    apk1 = 0.93 + 0.487118 * c14 * BdL**1.06806 * TdL**0.46106 * (L / L_R)**0.121563 * (L3dnabla)**0.36486 * (1 - C_P)**(-0.604247)

    S = L * (2 * T_M + B) * C_M**0.5 * (0.453 + 0.4425 * C_B - 0.2862 * C_M - 0.003467 * B / T_M + 0.3696 * C_WP) + 2.38 * A_BT / C_B  # área molhada

    # Resistência de atrito
    C_F = 0.075 / (np.log10(Rn) - 2)**2  # coeficiente de resistência de atrito

    R_F = 0.5 * rho * V_S**2 * S * C_F / 1000  # resistência de atrito [kN]

    # Resistência de apêndices
    R_APP = 0.5 * rho * V_S**2 * S_APP * apk2 * C_F / 1000  # resistência de apêndices [kN]

    # Resistência de ondas
    c3 = 0.56 * A_BT**1.5 / (B * T_M * (0.31 * A_BT**0.5 + T_F - h_B))
    c2 = np.exp(-1.89 * c3**0.5)
    c5 = 1 - 0.8 * A_T / (B * T_M)

    LdB = L / B
    if LdB < 12:
        lambda_value = 1.446 * C_P - 0.03 * LdB
    else:
        lambda_value = 1.446 * C_P - 0.36

    if L3dnabla < 512:
        c15 = -1.69385
    elif L3dnabla > 1726.9:
        c15 = 0
    else:
        c15 = -1.69385 + (L / nabla**(1/3) - 8) / 2.36

    m4 = c15 * 0.4 * np.exp(-0.034 * Fn**(-3.29))

    if Fn < 0.4:
        i_E = 1 + 89 * np.exp(-(LdB)**0.80856 * (1 - C_WP)**0.30484 * (1 - C_P - 0.0225 * LCB)** 0.6367 * (L_R / B)**0.34574 * (100 * nabla / L**3)**0.16302)
        
        if BdL < 0.11:
            c7 = 0.229577 * BdL**0.33333
        elif BdL > 0.25:
            c7 = 0.5 - 0.0625 * L / B
        else:
            c7 = BdL
        
        c1 = 2223105 * c7**3.78613 * (T_M / B)**1.07961 * (90 - i_E)**(-1.37565)

        if C_P < 0.8:
            c16 = 8.07981 * C_P - 13.8673 * C_P**2 + 6.984388 * C_P**3
        else:
            c16 = 1.73014 - 0.7067 * C_P

        m1 = 0.0140407 * L / T_M - 1.75254 * nabla**(1/3) / L - 4.79323 * B / L - c16

        R_W = c1 * c2 * c5 * nabla * rho * g * np.exp(m1 * Fn**d + m4 * np.cos(lambda_value * Fn**(-2))) / 1000

    elif Fn > 0.55:
        c17 = 6919.3 * C_M**(-1.3346) * (nabla / L**3)**2.00977 * (L / B - 2)**1.40692
        m3 = -7.2035 * (B / L)**0.326869 * (T_M / B)**0.605375

        R_W = c17 * c2 * c5 * nabla * rho * g * np.exp(m3 * Fn**d + m4 * np.cos(lambda_value * Fn**(-2))) / 1000

    else:
        m4 = c15 * 0.4 * np.exp(-0.034 * 0.4**(-3.29))
        i_E = 1 + 89 * np.exp(-(LdB)**0.80856 * (1 - C_WP)**0.30484 * (1 - C_P - 0.0225 * LCB) ** 0.6367 * (L_R / B)**0.34574 * (100 * nabla / L**3)**0.16302)
        
        if BdL < 0.11:
            c7 = 0.229577 * BdL**0.33333
        elif BdL > 0.25:
            c7 = 0.5 - 0.0625 * L / B
        else:
            c7 = BdL
        
        c1 = 2223105 * c7**3.78613 * (T_M / B)**1.07961 * (90 - i_E)**(-1.37565)

        if C_P < 0.8:
            c16 = 8.07981 * C_P - 13.8673 * C_P**2 + 6.984388 * C_P**3
        else:
            c16 = 1.73014 - 0.7067 * C_P

        m1 = 0.0140407 * L / T_M - 1.75254 * nabla**(1/3) / L - 4.79323 * B / L - c16

        R_WA = c1 * c2 * c5 * nabla * rho * g * np.exp(m1 * 0.4**d + m4 * np.cos(lambda_value * 0.4**(-2))) / 1000

        m4 = c15 * 0.4 * np.exp(-0.034 * 0.55**(-3.29))
        c17 = 6919.3 * C_M**(-1.3346) * (nabla / L**3)**2.00977 * (L / B - 2)**1.40692
        m3 = -7.2035 * (B / L)**0.326869 * (T_M / B)**0.605375

        R_WB = c17 * c2 * c5 * nabla * rho * g * np.exp(m3 * 0.55**d + m4 * np.cos(lambda_value * 0.55**(-2))) / 1000

        R_W = R_WA + (10 * Fn - 4) * (R_WB - R_WA) / 1.5

    # Resistência adicional devido à proa bulbosa
    P_B = 0.56 * (A_BT)**0.5 / (T_F - 1.5 * h_B)  # emergência da proa
    Fni = V_S / (g * (T_F - h_B - 0.25 * (A_BT)**0.5) + 0.15 * V_S**2)**0.5  # Número de Froude baseado na imersão
    # R_B = 0.11 * np.exp(-3 * P_B**(-2)) * Fni**3 * A_BT**1.5 * rho * g / (1 + Fni**2) / 1000

    # Resistência adicional de pressão devido ao espelho de água imerso
    FnT = 0
    if FnT < 5:
        c6 = 0.2 * (1 - 0.2 * FnT)
    else:
        c6 = 0

    R_TR = 0.5 * rho * V_S**2 * A_T * c6 / 1000

    # Correlação do modelo-navio para resistência
    T_FdL = T_F / L
    if T_FdL <= 0.04:
        c4 = T_FdL
    else:
        c4 = 0.04

    C_A = 0.006 * (L + 100)**(-0.16) - 0.00205 + 0.003 * (L / 7.5)**0.5 * C_B**4 * c2 * (0.04 - c4)
    R_A = 0.5 * rho * V_S**2 * S * C_A / 1000

    # Resistência total
    RtSim = np.array([0, 698.9, 2332, 7643, 22170])  # N
    Vsim = np.array([0, 0.514444, 1.54333, 2.57222, 3.858])  # m/s
    Rtotal = np.interp(V_S, Vsim, RtSim) / 1000  # kN

    return Rtotal, C_B, C_P, S, apk1, C_F, C_A, Fn, i_E

from evaluate_propeller import evaluate

V_S = 7
Z = 6
D = 0.8
AEdAO = 0.892
PdD = 0.671

P_B, strength,strengthMin, cavitation,cavitationMax, velocity,velocityMax = evaluate(V_S,D,Z,AEdAO,PdD)

print(f"{P_B=}")

cavitation_penalty = max(((cavitation - cavitationMax)/cavitationMax), 0) 
tip_velocity_penalty = max(((velocity - velocityMax)/velocityMax), 0)
strenght_penalty = max(((strengthMin - strength)/strengthMin), 0)

penalty = cavitation_penalty + tip_velocity_penalty + strenght_penalty

fit_value = P_B * (1 + penalty)

print("Fitness", fit_value)
print("Penalty", ("Yes" if penalty != 0 else "No"))
print("")

print('V_S', V_S)  
print('Z', Z)
print('D', D)
print('AEdAO', AEdAO)
print('PdD', PdD)
print('P_B', fit_value)
print('Strength', strength)
print('Strength_Min', strengthMin)
print('Cavitation', cavitation)
print('Cavitation_Max', cavitationMax)
print('Tip_Velocity', velocity)
print('Tip_Velocity_Max', velocityMax)
    
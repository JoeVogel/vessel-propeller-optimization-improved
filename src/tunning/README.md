## Instruções 

Necessário instalar o R e o pacote irace

## Comandos R

setwd("Desktop/Mestrado/hydrone-optimization/src/tunning/pso")

library("irace")

parameters <- readParameters(file = "parameters.txt")

scenario <- readScenario(filename = "scenario.txt", scenario = defaultScenario())

checkIraceScenario(scenario = scenario)

irace.main(scenario = scenario)
## Instruções 

Necessário instalar o R e o pacote irace

## Comandos R

linux:

    setwd("Desktop/Mestrado/hydrone-optimization/src/tunning/pso")

    library("irace")

    parameters <- readParameters(file = "parameters.txt")

    scenario <- readScenario(filename = "linux_scenario.txt", scenario = defaultScenario())

    checkIraceScenario(scenario = scenario)

    irace.main(scenario = scenario)

windows:

    setwd("C:\\Users\\Joe Vogel\\Desktop\\MESTRADO\\git\\hydrone-optimization\\src\\tunning\\cma")

    library("irace")

    parameters <- readParameters(file = "parameters.txt")

    scenario <- readScenario(filename = "windows_scenario.txt", scenario = defaultScenario())

    checkIraceScenario(scenario = scenario)

    irace.main(scenario = scenario)
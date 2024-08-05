#!/bin/bash

# Caminho para o ambiente virtual
VENV_PATH="/home/joe/Desktop/Mestrado/hydrone-optimization/.env"

# Ativa o ambiente virtual
source "$VENV_PATH/bin/activate"

# Executa o script Python com os parâmetros
python /home/joe/Desktop/Mestrado/hydrone-optimization/src/tunning/openai/target-runner.py "$@"
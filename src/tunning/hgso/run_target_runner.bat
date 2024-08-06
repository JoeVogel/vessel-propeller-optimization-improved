@echo off
REM Caminho para o ambiente virtual
set "VENV_PATH=C:\Users\Joe Vogel\Desktop\MESTRADO\git\hydrone-optimization\.venv"

REM Ativa o ambiente virtual
call "%VENV_PATH%\Scripts\activate.bat"

REM Executa o script Python com os parâmetros
python "C:\Users\Joe Vogel\Desktop\MESTRADO\git\hydrone-optimization\src\tunning\hgso\target-runner.py" %*
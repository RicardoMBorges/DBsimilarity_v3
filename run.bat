@echo on
setlocal EnableExtensions

REM --- Paths ---
set "script_dir=%~dp0"
set "venv_dir=%script_dir%dbsimilarity_env"
REM Use your real Python 3.11:
set "PY_EXE=C:\Users\borge\AppData\Local\Programs\Python\Python311\python.exe"

if not exist "%PY_EXE%" (
  echo Could not find Python 3.11 at: %PY_EXE%
  echo Install Python 3.11 or update PY_EXE above to the right path.
  pause & exit /b 1
)

REM --- Check requirements.txt ---
if not exist "%script_dir%requirements.txt" (
  echo requirements.txt not found in "%script_dir%"
  pause & exit /b 1
)

REM --- Create (or reuse) venv ---
if not exist "%venv_dir%\Scripts\activate" (
  echo Creating venv: "%venv_dir%"
  "%PY_EXE%" -m venv "%venv_dir%" || (echo venv creation failed & pause & exit /b 1)
)

call "%venv_dir%\Scripts\activate"

REM --- Install deps (use absolute paths) ---
"%venv_dir%\Scripts\python.exe" -m pip install --upgrade pip
"%venv_dir%\Scripts\python.exe" -m pip install -r "%script_dir%requirements.txt" || (
  echo Pip install failed. See notes below about RDKit on Windows.
  pause & exit /b 1
)

REM --- Jupyter + kernel ---
"%venv_dir%\Scripts\python.exe" -m pip install "notebook==6.5.2" "traitlets==5.9.0" ipykernel
"%venv_dir%\Scripts\python.exe" -m ipykernel install --user --name dbsimilarity_env --display-name "Python (dbsimilarity_env)"

REM --- Start Notebook from THIS env in THIS folder ---
pushd "%script_dir%"
"%venv_dir%\Scripts\python.exe" -m notebook
popd

pause

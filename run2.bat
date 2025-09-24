@echo on
setlocal EnableExtensions

REM --- Paths ---
set "script_dir=%~dp0"
set "venv_dir=%script_dir%dbsimilarity_env"

REM --- Locate Python 3.11 (portable) ---
set "PY_EXE="

REM 1) Typical per-user install (Windows Store/official installer)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
  set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)

REM 2) Use the Python launcher if available (py -3.11)
if not defined PY_EXE (
  where py >nul 2>&1 && for /f "delims=" %%I in ('
    py -3.11 -c "import sys;print(sys.executable)" 2^>nul
  ') do set "PY_EXE=%%I"
)

REM 3) Fall back to system "python" if it is >= 3.11
if not defined PY_EXE (
  where python >nul 2>&1 && for /f "delims=" %%I in ('
    python -c "import sys;print(sys.executable if sys.version_info[:2]>=(3,11) else \"\")" 2^>nul
  ') do if not "%%I"=="" set "PY_EXE=%%I"
)

REM 4) Common system-wide locations (just in case)
if not defined PY_EXE if exist "%ProgramFiles%\Python311\python.exe" set "PY_EXE=%ProgramFiles%\Python311\python.exe"
if not defined PY_EXE if exist "%ProgramFiles(x86)%\Python311\python.exe" set "PY_EXE=%ProgramFiles(x86)%\Python311\python.exe"

REM Final check
if not exist "%PY_EXE%" (
  echo Could not find Python 3.11. Please install it or add it to PATH.
  echo Searched: %%LOCALAPPDATA%%, py -3.11, PATH, and Program Files.
  pause & exit /b 1
)

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

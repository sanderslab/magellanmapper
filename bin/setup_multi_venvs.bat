@echo off
rem Set up Venv environments for multiple Python versions on Windows platforms

rem Usage:
rem   setup_venv.sh [env-dir]

rem Args:
rem   [env-dir]: Path to environment directory; defaults to "..\venvs\vmag".


rem parse user env directory path
set "venvs_dir=..\venvs"
if not "%~1" == "" (
  set "venvs_dir=%~1"
)
echo Setting the Venvs directory path to %venvs_dir%

pushd "%~dp0"
cd ..

rem specify full Python versions
for %%v in (3.6.8 3.7.7 3.8.7 3.9.1) do (
  echo Creating Venv for Python %%v
  call pyenv local "%%v"
  if exist "%venvs_dir%\py%%v"\ (
    echo Directory already exists, skipping
  ) else (
    call python -m venv "%venvs_dir%\py%%v""
  )
)

popd

rem keep window open if user double-clicked this script to launch it
pause

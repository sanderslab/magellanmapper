@echo off
rem Set up a Venv environment for MagellanMapper on Windows platforms

rem Usage:
rem   setup_venv.sh [env-dir]

rem Args:
rem   [env-dir]: Path to environment directory; defaulst to ..\venvs\vmag


rem parse user env directory path
set "venv_dir=..\venvs\vmag"
if not "%~1" == "" (
  set "venv_dir=%~1"
)
set "env_act=%venv_dir%\Scripts\Activate.bat"
echo Setting the Venv path to %venv_dir%

pushd "%~dp0"
cd ..

rem create new env if a Venv env does not already exist there
if exist "%venv_dir%\" (
  if not exist "%env_act%" (
    echo %venv_dir% exists but does not appear to be a venv."
    echo "Please choose another venv path. Exiting."
    exit /b 1
  )
  echo %venv_dir% Venv directory exist, will update...
) else (
  echo Creating a Venv environment in %venv_dir%
  call python -m venv "%venv_dir%"
)
if not exist "%env_act%" (
  echo Could not create environment at %env_act%, exiting
  exit /b 1
)

rem install/update app and all its dependencies
call "%env_act%"
call python -m pip install -U pip
call pip install --upgrade --upgrade-strategy eager -e .[all] --extra-index-url^
  https://pypi.fury.io/dd8/
echo Completed Venv environment setup! Please run %env_act%
echo to activate the environment if you open a new command prompt.

popd

rem keep window open if user double-clicked this script to launch it
pause

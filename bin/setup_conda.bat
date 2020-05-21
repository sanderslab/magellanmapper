:: Set up a Conda environment for MagellanMapper on Windows platforms
:: Author: David Young, 2020

@ECHO OFF

SET "conda_bit=x86"

pushd "%~dp0"
cd ..

IF "%CONDA_PREFIX%" == "" (
  ECHO Anaconda/Miniconda not found. If you have already installed it, please open the Anaconda Prompt from the Start Menu and rerun this script.
  ECHO Otherwise, we will attempt to install Miniconda.
  
  SET "install_conda=y"
  SET /p "install_conda=Download and install Miniconda? [y|n]: "
  IF NOT "%install_conda%" == "y" (
    ECHO Installation stopped, exiting.
    EXIT /b 1
  )
  
  reg query "HKLM\Software\Microsoft\Windows NT\CurrentVersion" /v "BuildLabEx" | >nul find /i ".x86fre." && set "os_bit=32" || set "os_bit=64"
  ECHO "%os_bit%"
  IF "%os_bit%" == "64" (
    SET "conda_bit=x86_64"
  )
  SET "conda_out=Miniconda3-latest-Windows-%conda_bit%.exe"
  bitsadmin /transfer downloadconda /download /priority FOREGROUND https://repo.anaconda.com/miniconda/%conda_out% "%cd%/%conda_out%"
  
  ECHO Downloaded Miniconda, about to install. Please allow the installer to initialize Miniconda for 'conda' to run.
  ".\%conda_out%"
  
  ECHO Please open the Anaconda Prompt from the Start Menu and rerun this script to continue installation of MagellanMapper.
  EXIT /b 0
)

ECHO Creating a Conda environment for MagellanMapper with all supporting packages
ECHO This may take awhile, especially during the 'Solving environment' and after the 'Executing transaction' steps
CALL conda env create -n mag -f environment.yml
conda env list | >nul find /i "mag" && set "check_env=0" || set "check_env=1"
IF "%check_env%" == "0" (
  ECHO MagellanMapper setup complete! Please run 'conda activate mag' before running MagellanMapper.
) ELSE (
  ECHO Conda environment could not be found. Consider retrying this setup.
)

popd

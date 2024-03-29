:: Build SimpleITK with Elastix support using MSVC
:: Author: David Young, 2019, 2022

:: Build SimpleITK with Elastix for Windows using MSVC build tools.
::
:: Compiled files will be found in ../../build_se relative to this script.
::
:: Usage:
::   build_se.bat [build-dir] [path-to-SimpleITK]
::
:: Args:
::   build-dir: Path to output build directory; defaults to "..\build_se".
::   path-to-SimpleITK: Path to existing SimpleITK directory or
::     where it will be stored; defaults to "..\SimpleITK".
::
:: Build requirements:
:: - MSVC C++ x64/x86 build tools (eg MSVC v142 VS 2019 C++; 2022 also works)
:: - Windows SDK (tested on 10.0.19041.0)
:: - Git (eg "Git for Windows" in VS2019)
:: - CMake >= 3.16.3 (official release or via MSVC now working)
:: - Run this script in a native tools prompt (eg "x64 Native Tools Command
::   Prompt to VS 2019")
::
:: Assumes:
:: - The SimpleITK git repo has been cloned to the parent
::   folder of this script's folder
:: - The Python path from an activated Conda environment will be used through
::   the CONDA_PREFIX environment variable

@ECHO off

:: parse arg for output build directory
SET "build_dir=build_se"
IF NOT "%~1" == "" (
  SET "build_dir=%~1"
)
call :get_abs_path "%build_dir%"
SET "build_dir=%abs_path%"
ECHO Setting the build directory path to %build_dir%

:: parse arg for SimpleITK, converting to an absolute path
SET "se_dir=..\SimpleITK"
IF NOT "%~2" == "" (
  SET "se_dir=%~2"
)
call :get_abs_path "%se_dir%"
SET "se_dir=%abs_path%"
ECHO Setting the SimpleITK path to %se_dir%

:: set up Cmake
SET "cmake=cmake.exe"
SET "generator=Visual Studio 17 2022"
SET "arch=x64"
SET "python_exe="

IF NOT "%CONDA_PREFIX%" == "" (
  :: use Python in the current Conda environment
  SET "python_exe=-DPYTHON_EXECUTABLE="%CONDA_PREFIX%\python.exe""
)

:: discover Python include directory from distutils
FOR /f "tokens=*" %%g IN ('python -c "from distutils.sysconfig import get_python_inc; import os; print(os.path.dirname(get_python_inc()))"') DO (set "python_ver_path=%%g")
SET "python_incl_dir=%python_ver_path%\include"
SET "python_incl_arg=-DPYTHON_INCLUDE_DIR=%python_incl_dir%"

:: parse Python library
FOR /f "tokens=*" %%g IN ('python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"') DO (set var=%%g)
SET "python_lib_path=%python_ver_path%\libs\python%var%.lib"
SET "python_lib_arg="
IF EXIST "%python_lib_path%" (
  SET "python_lib_arg=-DPYTHON_LIBRARY=%python_lib_path%"
)
ECHO Set Python include dir arg to: %python_incl_arg%
ECHO Set Python library arg to: %python_lib_arg%

:: make build folder
pushd "%~dp0"
mkdir "%build_dir%"
cd "%build_dir%"

:: configure with CMake to generate only a Python wrapper, running twice
:: since some paths may not be exposed until the 2nd run
"%cmake%" -G "%generator%" -A "%arch%" -T host="%arch%"^
 "%python_exe%" "%python_incl_arg%" "%python_lib_arg%"^
 -DWRAP_PYTHON:BOOL=ON -DWRAP_JAVA:BOOL=OFF^
 -DWRAP_LUA:BOOL=OFF -DWRAP_R:BOOL=OFF -DWRAP_RUBY:BOOL=OFF^
 -DWRAP_TCL:BOOL=OFF -DWRAP_CSHARP:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF^
 -DBUILD_TESTING:BOOL=OFF -DSimpleITK_PYTHON_USE_VIRTUALENV:BOOL=OFF^
 -DSimpleITK_USE_ELASTIX=ON^
 "%se_dir%\SuperBuild"
"%cmake%" -G "%generator%" -A "%arch%" -T host="%arch%"^
 "%se_dir%\SuperBuild"

:: build SimpleITK
msbuild ALL_BUILD.vcxproj /p:Configuration=Release

:: build Python source and wheel packages for distribution
cd SimpleITK-build/Wrapping/Python
python setup.py build sdist bdist_wheel

popd

exit /b

:: Convert a path to an absolute path.
:get_abs_path
  set "abs_path=%~f1
  exit /b

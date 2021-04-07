:: Build SimpleElastix with MSVC
:: Author: David Young, 2019, 2020

:: Build SimpleElastix for Windows using MSVC build tools. Compiled files
:: will be found in ../../build_se relative to this script.
::
:: Usage:
::   build_se.bat [build-dir] [path-to-SimpleElastix]
::
:: Args:
::   build-dir: Path to output build directory.
::   path-to-SimpleElastix: Path to existing SimpleElastix directory or
::     where it will be stored.
::
:: MSVC requirements:
:: - MSVC C++ x64/x86 build tools (tested on MSVC v142 VS 2019 C++ with the
::   "Latest" version checked)
:: - C++ CMake Tools for Windows
:: - Git (eg "Git for Windows")
:: - Run this script in an "x64 Native Tools Command Prompt to VS 2019"
::
:: Assumes:
:: - The SimpleElastix git repo has been cloned to the parent
::   folder of this script's folder
:: - The Python path from an activated Conda environment will be used through
::   the CONDA_PREFIX environment variable
:: Tested with CMake 3.15; has *not* worked with MSVS CMake 3.14 on our testing.

:: parse arg for output build directory
SET "build_dir=build_se"
IF NOT "%~1" == "" (
  SET "build_dir=%~1"
)
ECHO Setting the build directory path to %build_dir%

:: parse arg for SimpleElastix, converting to an absolute path
SET "se_dir=..\SimpleElastix"
IF NOT "%~2" == "" (
  SET "se_dir=%~2"
)
call :get_abs_path "%se_dir%"
SET "se_dir=%abs_path%"
ECHO Setting the SimpleElastix path to %se_dir%

:: set up Cmake
SET "cmake=cmake.exe"
SET "generator=Visual Studio 16 2019"
SET "arch=x64"
SET "python_exe="

IF NOT "%CONDA_PREFIX%" == "" (
  :: use Python in the current Conda environment
  SET "python_exe=-DPYTHON_EXECUTABLE="%CONDA_PREFIX%\python.exe""
)

:: discover Python include directory from distutils
FOR /f "tokens=*" %%g IN ('python -c "from distutils.sysconfig import get_python_inc; import os; print(os.path.dirname(get_python_inc()))"') DO (set var=%%g)
SET "python_ver_path=%var%"
SET "python_incl_dir=%python_ver_path%\include"
SET "python_incl_arg=-DPYTHON_INCLUDE_DIR=%python_incl_dir%"

:: parse Python library
for /F "delims=" %%i in (%python_ver_path%) do set dirname="%%~dpi"
SET "python_ver_majmin=%python_ver_path:~-5,-2%"
SET "python_ver_majmin=%python_ver_majmin:.=%"
:: TODO: the .lib file may be sufficient
SET "python_lib_path_a=%python_ver_path%\libs\libpython%python_ver_majmin%.a"
SET "python_lib_path_lib=%python_ver_path%\libs\python%python_ver_majmin%.lib"
SET "python_lib_arg="
IF EXIST "%python_lib_path_a%" (
  SET "python_lib_arg=-DPYTHON_LIBRARY=%python_lib_path_a%;%python_lib_path_lib%"
)
ECHO Set Python include dir arg to: %python_incl_arg%
ECHO Set Python library arg to: %python_lib_arg%

:: make build folder
pushd "%~dp0"
cd ../..
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
 "%se_dir%\SuperBuild"
"%cmake%" -G "%generator%" -A "%arch%" -T host="%arch%"^
 "%se_dir%\SuperBuild"

:: build SimpleElastix
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

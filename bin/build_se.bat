:: Build SimpleElastix with MSVC
:: Author: David Young, 2019, 2020

:: Build SimpleElastix for Windows using MSVC build tools.
:: Assumes:
:: - The SimpleElastix git repo has been cloned to the parent
::   folder of this script's folder
:: - The CONDA_PREFIX environment variable will be used for the Python path
:: Tested with CMake 3.15; has *not* worked with MSVS CMake 3.14 on our testing.

SET "cmake=cmake.exe"
SET "build_dir=build_se"
SET "generator=Visual Studio 16 2019"
SET "arch=x64"

:: make build folder
pushd "%~dp0"
cd ../..
mkdir "%build_dir%"
cd "%build_dir%"

:: configure with CMake to generate only a Python wrapper, rerunning
:: since some paths may not be exposed until the 2nd run
"%cmake%" -G "%generator%" -A "%arch%" -T host="%arch%" -DWRAP_JAVA:BOOL=OFF^
 -DPYTHON_EXECUTABLE="%CONDA_PREFIX%\python.exe"^
 -DWRAP_LUA:BOOL=OFF -DWRAP_R:BOOL=OFF -DWRAP_RUBY:BOOL=OFF -DWRAP_TCL:BOOL=OFF^
 -DWRAP_CSHARP:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF -DBUILD_TESTING:BOOL=OFF^
 -DSimpleITK_PYTHON_USE_VIRTUALENV:BOOL=OFF ..\SimpleElastix\SuperBuild
"%cmake%" -G "%generator%" -A "%arch%" -T host="%arch%" ..\SimpleElastix\SuperBuild

: build SimpleElastix
msbuild ALL_BUILD.vcxproj /p:Configuration=Release
popd

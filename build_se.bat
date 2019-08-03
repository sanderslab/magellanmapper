:: Build SimpleElastix with MSVC
:: Author: David Young, 2019

: Build SimpleElastix for Windows using MSVC build tools.
: Assumes that the SimpleElastix git repo has been cloned to the parent 
: folder of this script

: make build folder
cd ..
mkdir build_se
cd build_se

: configure with CMake to generate only a Python wrapper, rerunning 
: since some paths may not be exposed until the 2nd run
cmake -G "Visual Studio 16 2019" -A x64 -T host=x64 -DWRAP_JAVA:BOOL=OFF^
 -DWRAP_LUA:BOOL=OFF -DWRAP_R:BOOL=OFF -DWRAP_RUBY:BOOL=OFF -DWRAP_TCL:BOOL=OFF^
 -DWRAP_CSHARP:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF -DBUILD_TESTING:BOOL=OFF^
 -DSimpleITK_PYTHON_USE_VIRTUALENV:BOOL=OFF ..\SimpleElastix\SuperBuild
cmake.exe -G "Visual Studio 16 2019" -A x64 -T host=x64 ..\SimpleElastix\SuperBuild

: build SimpleElastix
msbuild ALL_BUILD.vcxproj /p:Configuration=Release

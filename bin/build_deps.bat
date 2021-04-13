@echo off
rem Build custom binaries for MagellanMapper dependencies on Windows platforms

rem Usage:
rem   setup_venv.sh [envs-dir] [output-dir]

rem Args:
rem   [env-dir]: Path to environment directory; defaults to `..\venvs\vmag`.
rem   [output-dir]: Path to output directory; defaults to `..\build_deps`.


rem parse user arg for path containing Venvs
set "venvs_dir=..\venvs"
if not "%~1" == "" (
  set "venvs_dir=%~1"
)
echo Setting the Venvs directory path to %venvs_dir%

rem parse user env directory path
set "output_dir=..\build_deps"
if not "%~2" == "" (
  set "output_dir=%~2"
)
echo Setting the output directory path to %output_dir%

pushd "%~dp0"
cd ..

rem build binaries within each Python version Venv environment
for %%v in (3.6.8 3.7.7 3.8.7 3.9.1) do (
  echo Activating Venv for py%%v
  call "%venvs_dir%\py%%v\Scripts\Activate.bat"
  call pip install wheel
  
  rem build SimpleElastix
  call bin\build_se.bat "%output_dir%\build_se_py%%v"
  call copy "%output_dir%\build_se_py%%v\SimpleITK-build\Wrapping\Python\dist\"*.whl "%output_dir%"
  call copy "%output_dir%\build_se_py%%v\SimpleITK-build\Wrapping\Python\dist\"*.tar.gz "%output_dir%"
  
  rem build Javabridge; Numpy 1.19 is latest ver for Python 3.6
  pip install cython
  pip install numpy~=1.19
  call bin\build_jb.bat "%output_dir%"
  
  call deactivate
)

popd

rem keep window open if user double-clicked this script to launch it
pause

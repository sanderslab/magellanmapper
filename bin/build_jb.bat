rem Build Python-Javabridge with MSVC and OpenJDK on Windows

rem Build Python-Javabridge wheels for Windows using MSVC build tools.
rem
rem Usage:
rem   build_jb.bat [build-dir] [path-to-Javabridge]
rem
rem Args:
rem   build-dir: Path to output build directory; defaults to "build_jb".
rem   path-to-Javabridge: Path to existing Javabridge directory or
rem     where it will be stored; defaults to "..\python-javabridge".

rem parse arg for output build directory
set "build_dir=build_jb"
if not "%~1" == "" (
  set "build_dir=%~1"
)
call :get_abs_path "%build_dir%"
set "build_dir=%abs_path%"
echo Setting the build directory path to %build_dir%

rem parse arg for SimpleElastix, converting to an absolute path
set "jb_dir=..\python-javabridge"
if not "%~2" == "" (
  set "jb_dir=%~2"
)
call :get_abs_path "%jb_dir%"
set "jb_dir=%abs_path%"
echo Setting the Javabridge path to %jb_dir%

rem Javabridge on Windows requires JDK_HOME, not JAVA_HOME (though may want
rem to set JAVA_HOME to same path and point PATH to JAVA_HOME\bin just to
rem be safe)
if "%JDK_HOME%" == "" (
  echo JDK_HOME not found, required for Javabridge build, exiting
  exit /b 1
)

rem get and/or enter git repo
pushd "%~dp0"
if not exist "%jb_dir%\" (
  echo Cloning Python-Javabridge git repo to %jb_dir%...
  git clone https://github.com/LeeKamentsky/python-javabridge.git "%jb_dir%"
)
if not exist "%jb_dir%\" (
  echo Could not create %jb_dir%, existing
  exit /b 1
)
cd "%jb_dir%"

rem restore repo dir to pristine state
git clean -dfx

rem build binaries, wheel, and source distribution
echo "Building Python-Javabridge"
python setup.py build
python setup.py bdist_wheel
python setup.py sdist

rem make build folder
if not exist "%build_dir%\" (
  mkdir "%build_dir%"
)
copy dist\*.whl "%build_dir%"
copy dist\*.tar.gz "%build_dir%"

popd

exit /b

rem Convert a path to an absolute path.
:get_abs_path
  set "abs_path=%~f1
  exit /b

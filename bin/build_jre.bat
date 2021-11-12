@echo on
@rem Extract a distributable Java Runtime Environment for MagellanMapper
@rem dependencies

@rem Usage:
@rem   build_jre.bat [env-dir]

@rem Args:
@rem   [env-dir]: Path to environment directory; defaults to ..\venvs\vmag


@rem parse user env directory path
set "venv_dir=..\venvs\vmag"
if not "%~1" == "" (
  set "venv_dir=%~1"
)
echo Setting the Venv path to %venv_dir%

@rem get dependencies, storing in variable "var"
for /f "tokens=*" %%g in ('jdeps --print-module-deps --recursive^
 --ignore-missing-deps -q^
 "%venv_dir%"\Lib\site-packages\bioformats\jars\bioformats_package.jar') do ^
 (set var=%%g)

@rem extract JRE to distributable for the discovered dependencies
jlink --no-header-files --no-man-pages --compress=2^
 --strip-java-debug-attributes --add-modules %var% --output jre_windows

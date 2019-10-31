@ECHO OFF
@setlocal enableextensions enabledelayedexpansion

CLS

TITLE Contact angle analysis script
REM Simple bash script that runs contact angle analysis on all files in a folder

REM Check for common anaconda installation locations
CALL "C:\Users\%USERNAME%\Miniconda3\Scripts\activate.bat" "C:\Users\%USERNAME%\Miniconda3\envs\contact_angles"

REM Check that a target directory was specified
SET dirpath=%~dp1
IF NOT EXIST "%dirpath%" (
	ECHO %~n0: direcotry not found - %dirpath% >&2
	CALL "C:\Users\%USERNAME%\Miniconda3\Scripts\deactivate.bat"
	EXIT /B 1
)

REM Process any other parameters on the command line by first shifting out the directory
SET params = ""
:Loop
	SHIFT
	IF "%1"=="" GOTO Continue
	SET params=!params! %1
	GOTO Loop
:Continue

REM Now loop through all the files that haven't been run yet and process them
FOR %%f IN ("%dirpath%*.avi") DO (
	SET folder=%%~dpf
	SET file=%%~nxf

	IF NOT EXIST "%folder%results_%file%.csv" (
		ECHO "Running on %%f"
		CALL python analysis.py "%%f" %params%
	)
)

CALL "C:\Users\%USERNAME%\Miniconda3\Scripts\deactivate.bat"
endlocal
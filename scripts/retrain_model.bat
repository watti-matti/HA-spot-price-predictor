@echo off
REM ============================================================
REM  HA Spot Price Predictor - one-shot PC retraining script
REM
REM  Retrains EVERYTHING on your PC (fast) so your Home Assistant
REM  host (e.g. a Raspberry Pi) never has to. It refits both:
REM    * the base hourly + D(k) duration model, and
REM    * the pipeline layers (L1 seasonal, L2-L4 spike, solar),
REM  then stages the 4 updated JSON artifacts ready to copy to HA.
REM
REM  Run it by double-clicking, or from a terminal. It finds the
REM  repo from its own location - keep it in the scripts\ folder.
REM  Requires Python 3.11+ and the repo checked out on this PC.
REM ============================================================

setlocal EnableExtensions

REM ======================= EDIT ME ===========================
REM Fingrid API key (free, https://data.fingrid.fi). OPTIONAL:
REM leave BLANK to train without nuclear features and skip the
REM solar sub-model. The model still works fully without it.
set "FINGRID_API_KEY="

REM Region config file under config\regions\<REGION>.yaml
set "REGION=finland"

REM Years of price history to train on (4 = recommended).
set "YEARS=4"

REM Python launcher. Use a venv if you have one, e.g.
REM   set "PYTHON=C:\path\to\venv\Scripts\python.exe"
set "PYTHON=python"

REM Install/upgrade Python dependencies first? 1=yes 0=no
set "INSTALL_DEPS=1"

REM Run the training unit tests at the end? 1=yes 0=no
set "RUN_TESTS=1"
REM ===========================================================

pushd "%~dp0.." || (echo ERROR: could not locate repo root & exit /b 1)
echo Repo root: %CD%
echo.

set "FK="
if not "%FINGRID_API_KEY%"=="" set "FK=--fingrid-key %FINGRID_API_KEY%"
if "%FINGRID_API_KEY%"=="" echo [note] No Fingrid key - nuclear features off, solar sub-model skipped.
echo.

if "%INSTALL_DEPS%"=="1" (
  echo === [1/5] Installing Python dependencies ===
  "%PYTHON%" -m pip install -r requirements.txt scikit-learn pytest || goto :fail
  echo.
)

echo === [2/5] Training base model: fetch data + hourly Ridge + duration D-of-k ===
REM Auto-retry: prices + each weather location are cached on disk, so a
REM transient timeout only re-fetches what failed - not from scratch.
set "ATTEMPT=0"
:train_retry
set /a ATTEMPT+=1
"%PYTHON%" -m src.train_model --region %REGION% %FK% --years %YEARS%
if not errorlevel 1 goto :train_ok
if %ATTEMPT% GEQ 3 goto :fail
echo [retry] training step failed - attempt %ATTEMPT% of 3. Cached data reused; retrying in 20s...
timeout /t 20 /nobreak >nul
goto :train_retry
:train_ok
echo.

echo === [3/5] Retraining pipeline layers: seasonal + spike + solar ===
REM Run retrain.py as a FILE (not -m): the package __init__ imports
REM Home Assistant, which is not installed on a training PC.
"%PYTHON%" custom_components\spot_price_predictor\retrain.py --layers seasonal spike solar %FK% || goto :fail
echo.

echo === [4/5] Staging updated artifacts ===
set "DATA=custom_components\spot_price_predictor\data"
copy /Y "output\model_coefs.json" "%DATA%\model_coefs_default.json" >nul || goto :fail
if not exist "output\ha_deploy" mkdir "output\ha_deploy"
copy /Y "%DATA%\model_coefs_default.json"         "output\ha_deploy\" >nul
copy /Y "%DATA%\seasonal_components_default.json"  "output\ha_deploy\" >nul
copy /Y "%DATA%\spike_model_default.json"          "output\ha_deploy\" >nul
copy /Y "%DATA%\solar_submodel_default.json"       "output\ha_deploy\" >nul 2>nul
echo Updated artifacts staged in: %CD%\output\ha_deploy
echo.

if "%RUN_TESTS%"=="1" (
  echo === [5/5] Running training tests ===
  "%PYTHON%" -m pytest tests\test_training.py -q || goto :fail
  echo.
)

echo ============================================================
echo  DONE. To update your Home Assistant host:
echo.
echo    Copy the 3-4 JSON files from
echo        %CD%\output\ha_deploy\
echo    to your HA host at
echo        ^<config^>/custom_components/spot_price_predictor/data/
echo    then restart Home Assistant (or reload the integration).
echo.
echo  Tip: for the base model only you can skip the restart and use
echo  the spot_price_predictor.upload_coefficients service with
echo  file_path set to model_coefs_default.json on the HA host.
echo ============================================================
popd
endlocal
exit /b 0

:fail
echo.
echo *** Retraining FAILED - see the error above. Some artifacts may be
echo *** partially updated; do NOT copy output\ha_deploy to HA until a
echo *** clean run completes. ***
popd
endlocal
exit /b 1

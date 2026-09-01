@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0"
pushd "%PROJECT_ROOT%" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Could not enter project folder: %PROJECT_ROOT%
    pause
    exit /b 1
)

rem ------------------------------------------------------------
rem Hugging Face cache configuration
rem ------------------------------------------------------------
rem This must point at the cache that actually has the models downloaded
rem (E:\AI_Cache\huggingface_cache\hub) -- NOT a fresh/empty folder, or every
rem model will look "not cached" and this script will try to download them
rem all again.
set "HF_HOME=E:\AI_Cache\huggingface_cache"
set "HF_HUB_CACHE=%HF_HOME%\hub"

rem Your current Windows setup does not permit symlinks, so Hugging Face
rem automatically uses ordinary cached file copies. This setting only hides
rem the harmless warning; it does not change model files or evaluation.
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

rem Ensure download progress bars are not disabled by an inherited setting.
set "HF_HUB_DISABLE_PROGRESS_BARS=0"

if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%"

rem ------------------------------------------------------------
rem Locate Python
rem ------------------------------------------------------------
set "PYTHON_EXE="

if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%PROJECT_ROOT%venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%PROJECT_ROOT%env\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%env\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%PROJECT_ROOT%nli_env\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%nli_env\Scripts\python.exe"

if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
)

if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
)

set "CACHE_SCRIPT=%PROJECT_ROOT%cache_one_nli_model.py"
set "HELPER_PS1=%PROJECT_ROOT%run_and_log_visible_progress.ps1"
set "LOG_DIR=%PROJECT_ROOT%outputs\run_logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"

set "MASTER_LOG=%LOG_DIR%\nli_visible_run_%STAMP%.log"
set "SUMMARY_FILE=%LOG_DIR%\nli_visible_run_%STAMP%_summary.csv"

rem Each run writes predictions/reports/confusion-matrices to its own fresh,
rem timestamped folder instead of outputs\<model>\, so a rerun never
rem overwrites a prior run's results. Run logs still go to outputs\run_logs
rem as before (they are always uniquely timestamped and never overwrite).
set "NLI_OUTPUT_ROOT=%PROJECT_ROOT%outputs_rerun_%STAMP%"
if not exist "%NLI_OUTPUT_ROOT%" mkdir "%NLI_OUTPUT_ROOT%"

(
echo ClearClause Sequential NLI Download and Evaluation
echo Start: %DATE% %TIME%
echo Project root: %PROJECT_ROOT%
echo Python: %PYTHON_EXE%
echo HF_HOME: %HF_HOME%
echo HF_HUB_CACHE: %HF_HUB_CACHE%
echo Output root for this run: %NLI_OUTPUT_ROOT%
echo.
) > "%MASTER_LOG%"

echo model,download_status,download_exit,evaluation_status,evaluation_exit,output_folder>>"%SUMMARY_FILE%"

if not defined PYTHON_EXE (
    echo ERROR: No usable Python installation was found.
    echo ERROR: No usable Python installation was found.>>"%MASTER_LOG%"
    popd
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo ERROR: Detected Python executable does not exist:
    echo %PYTHON_EXE%
    echo ERROR: Invalid Python path: %PYTHON_EXE%>>"%MASTER_LOG%"
    popd
    pause
    exit /b 1
)

if not exist "%CACHE_SCRIPT%" (
    echo ERROR: Per-model cache script was not found:
    echo %CACHE_SCRIPT%
    echo ERROR: Cache script missing: %CACHE_SCRIPT%>>"%MASTER_LOG%"
    popd
    pause
    exit /b 1
)

if not exist "%HELPER_PS1%" (
    echo ERROR: PowerShell logging helper was not found:
    echo %HELPER_PS1%
    echo ERROR: Logging helper missing: %HELPER_PS1%>>"%MASTER_LOG%"
    popd
    pause
    exit /b 1
)

echo Using Python:
echo %PYTHON_EXE%
echo.
echo Hugging Face cache:
echo %HF_HUB_CACHE%
echo.
echo Results for this run will be written to:
echo %NLI_OUTPUT_ROOT%
echo.
echo Download progress will appear below.
echo Python selected: %PYTHON_EXE%>>"%MASTER_LOG%"

"%PYTHON_EXE%" -c "import torch, transformers, pandas, sklearn" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Required packages are missing from:
    echo %PYTHON_EXE%
    echo Required: torch, transformers, pandas, scikit-learn
    echo ERROR: Required packages missing from selected Python.>>"%MASTER_LOG%"
    popd
    pause
    exit /b 1
)

set /a TOTAL_MODELS=0
set /a DOWNLOAD_SUCCESS=0
set /a DOWNLOAD_FAILED=0
set /a EVAL_SUCCESS=0
set /a EVAL_FAILED=0
set /a EVAL_SKIPPED=0
set /a MISSING_OUTPUTS=0

call :PROCESS_MODEL "RoBERTa-large-MNLI" "FacebookAI/roberta-large-mnli" "evaluation_scripts\evaluate_roberta_large_mnli.py" "roberta_large_mnli"
call :PROCESS_MODEL "DeBERTa-v3-base" "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli" "evaluation_scripts\evaluate_deberta_v3_base.py" "deberta_v3_base"
call :PROCESS_MODEL "ModernBERT-base-NLI" "tasksource/ModernBERT-base-nli" "evaluation_scripts\evaluate_modernbert_nli.py" "modernbert_nli"
call :PROCESS_MODEL "DeBERTa-v3-small" "cross-encoder/nli-deberta-v3-small" "evaluation_scripts\evaluate_deberta_baseline.py" "deberta_v3_small"
call :PROCESS_MODEL "DistilRoBERTa-NLI" "cross-encoder/nli-distilroberta-base" "evaluation_scripts\evaluate_distilroberta_nli.py" "distilroberta_nli"

echo.
echo ============================================================
echo FINAL SUMMARY
echo ============================================================
echo Total models: %TOTAL_MODELS%
echo Downloads successful: %DOWNLOAD_SUCCESS%
echo Downloads failed: %DOWNLOAD_FAILED%
echo Evaluations successful: %EVAL_SUCCESS%
echo Evaluations failed: %EVAL_FAILED%
echo Evaluations skipped after download failure: %EVAL_SKIPPED%
echo Evaluations with missing outputs: %MISSING_OUTPUTS%
echo Master log: %MASTER_LOG%
echo Summary CSV: %SUMMARY_FILE%

(
echo.
echo Final summary
echo End: %DATE% %TIME%
echo Total models: %TOTAL_MODELS%
echo Downloads successful: %DOWNLOAD_SUCCESS%
echo Downloads failed: %DOWNLOAD_FAILED%
echo Evaluations successful: %EVAL_SUCCESS%
echo Evaluations failed: %EVAL_FAILED%
echo Evaluations skipped after download failure: %EVAL_SKIPPED%
echo Evaluations with missing outputs: %MISSING_OUTPUTS%
echo Summary CSV: %SUMMARY_FILE%
) >> "%MASTER_LOG%"

popd
echo.
pause
exit /b 0

:PROCESS_MODEL
set "MODEL_DISPLAY=%~1"
set "MODEL_ID=%~2"
set "SCRIPT_REL=%~3"
set "OUTPUT_NAME=%~4"

set "SCRIPT_PATH=%PROJECT_ROOT%%SCRIPT_REL%"
set "OUTPUT_DIR=%NLI_OUTPUT_ROOT%\%OUTPUT_NAME%"
set "DOWNLOAD_LOG=%LOG_DIR%\%OUTPUT_NAME%_download_%STAMP%.log"
set "EVAL_LOG=%LOG_DIR%\%OUTPUT_NAME%_evaluation_%STAMP%.log"

set /a TOTAL_MODELS+=1

echo.
echo ============================================================
echo MODEL: %MODEL_DISPLAY%
echo STEP 1: Download/cache checkpoint
echo Hugging Face ID: %MODEL_ID%
echo Download log: %DOWNLOAD_LOG%
echo ============================================================

(
echo.
echo ============================================================
echo Model: %MODEL_DISPLAY%
echo Hugging Face ID: %MODEL_ID%
echo Download start: %DATE% %TIME%
echo Download log: %DOWNLOAD_LOG%
echo ============================================================
) >> "%MASTER_LOG%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%HELPER_PS1%" ^
    -PythonExe "%PYTHON_EXE%" ^
    -ScriptPath "%CACHE_SCRIPT%" ^
    -LogPath "%DOWNLOAD_LOG%" ^
    -ScriptArgs "%MODEL_ID%"

set "DOWNLOAD_EXIT=!ERRORLEVEL!"

if not "!DOWNLOAD_EXIT!"=="0" (
    echo.
    echo DOWNLOAD FAILED: %MODEL_DISPLAY% returned exit code !DOWNLOAD_EXIT!
    echo Evaluation for this model will be skipped.
    echo Download failed with exit code !DOWNLOAD_EXIT!. Evaluation skipped.>>"%MASTER_LOG%"
    echo "%MODEL_DISPLAY%",FAILED,!DOWNLOAD_EXIT!,SKIPPED,,"%OUTPUT_DIR%">>"%SUMMARY_FILE%"
    set /a DOWNLOAD_FAILED+=1
    set /a EVAL_SKIPPED+=1
    goto :eof
)

set /a DOWNLOAD_SUCCESS+=1
echo Download/cache successful for %MODEL_DISPLAY%.>>"%MASTER_LOG%"

echo.
echo ============================================================
echo MODEL: %MODEL_DISPLAY%
echo STEP 2: Run evaluation
echo Script: %SCRIPT_PATH%
echo Evaluation log: %EVAL_LOG%
echo ============================================================

if not exist "%SCRIPT_PATH%" (
    echo ERROR: Evaluation script not found: %SCRIPT_PATH%
    echo Evaluation script missing: %SCRIPT_PATH%>>"%MASTER_LOG%"
    echo "%MODEL_DISPLAY%",SUCCESS,0,FAILED_SCRIPT_NOT_FOUND,2,"%OUTPUT_DIR%">>"%SUMMARY_FILE%"
    set /a EVAL_FAILED+=1
    goto :eof
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%HELPER_PS1%" ^
    -PythonExe "%PYTHON_EXE%" ^
    -ScriptPath "%SCRIPT_PATH%" ^
    -LogPath "%EVAL_LOG%"

set "EVAL_EXIT=!ERRORLEVEL!"

if not "!EVAL_EXIT!"=="0" (
    echo.
    echo EVALUATION FAILED: %MODEL_DISPLAY% returned exit code !EVAL_EXIT!
    echo Evaluation failed with exit code !EVAL_EXIT!.>>"%MASTER_LOG%"

    if exist "%OUTPUT_DIR%" (
        for %%F in ("%OUTPUT_DIR%\*_overlength_rows.csv") do (
            if exist "%%~fF" (
                echo Overlength report: %%~fF
                echo Overlength report: %%~fF>>"%MASTER_LOG%"
            )
        )
    )

    echo "%MODEL_DISPLAY%",SUCCESS,0,FAILED,!EVAL_EXIT!,"%OUTPUT_DIR%">>"%SUMMARY_FILE%"
    set /a EVAL_FAILED+=1
    goto :eof
)

set "MODEL_MISSING=0"

call :CHECK_FILE "%OUTPUT_DIR%\validation_predictions.csv"
call :CHECK_FILE "%OUTPUT_DIR%\validation_report.txt"
call :CHECK_FILE "%OUTPUT_DIR%\validation_confusion_matrix.csv"
call :CHECK_FILE "%OUTPUT_DIR%\internal_test_predictions.csv"
call :CHECK_FILE "%OUTPUT_DIR%\internal_test_report.txt"
call :CHECK_FILE "%OUTPUT_DIR%\internal_test_confusion_matrix.csv"
call :CHECK_FILE "%OUTPUT_DIR%\external_test_predictions.csv"
call :CHECK_FILE "%OUTPUT_DIR%\external_test_report.txt"
call :CHECK_FILE "%OUTPUT_DIR%\external_test_confusion_matrix.csv"
call :CHECK_FILE "%OUTPUT_DIR%\heldout_combined_predictions.csv"
call :CHECK_FILE "%OUTPUT_DIR%\heldout_combined_report.txt"
call :CHECK_FILE "%OUTPUT_DIR%\heldout_combined_confusion_matrix.csv"
call :CHECK_FILE "%OUTPUT_DIR%\summary_metrics.csv"

if "!MODEL_MISSING!"=="0" (
    echo SUCCESS: %MODEL_DISPLAY% downloaded and evaluated successfully.
    echo Download and evaluation completed successfully.>>"%MASTER_LOG%"
    echo "%MODEL_DISPLAY%",SUCCESS,0,SUCCESS,0,"%OUTPUT_DIR%">>"%SUMMARY_FILE%"
    set /a EVAL_SUCCESS+=1
) else (
    echo WARNING: %MODEL_DISPLAY% evaluation returned success but outputs are missing or empty.
    echo Evaluation completed with missing outputs.>>"%MASTER_LOG%"
    echo "%MODEL_DISPLAY%",SUCCESS,0,COMPLETED_WITH_MISSING_OUTPUTS,0,"%OUTPUT_DIR%">>"%SUMMARY_FILE%"
    set /a MISSING_OUTPUTS+=1
)

goto :eof

:CHECK_FILE
set "CHECK_PATH=%~1"

if not exist "%CHECK_PATH%" (
    echo Missing file: %CHECK_PATH%
    echo Missing file: %CHECK_PATH%>>"%MASTER_LOG%"
    set "MODEL_MISSING=1"
    goto :eof
)

for %%Z in ("%CHECK_PATH%") do (
    if %%~zZ LEQ 0 (
        echo Empty file: %CHECK_PATH%
        echo Empty file: %CHECK_PATH%>>"%MASTER_LOG%"
        set "MODEL_MISSING=1"
    )
)
goto :eof

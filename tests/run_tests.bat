@echo off
set "PYTHONPATH=%~dp0.."
python -m pytest tests/test_integration.py -v
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Tests FAILED!
    exit /b %ERRORLEVEL%
) else (
    echo.
    echo ✅ All tests PASSED!
)

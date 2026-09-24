@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set "SETUP_EXIT_CODE=%ERRORLEVEL%"
if not "%SETUP_EXIT_CODE%"=="0" (
    echo.
    echo Setup failed. Please read the error above and try again.
    pause
    exit /b %SETUP_EXIT_CODE%
)
echo.
echo Setup complete. Double-click start.cmd to run FocusLens.
pause
exit /b 0

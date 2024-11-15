@echo off
echo AutoClipper needs TwitchDownloaderCLI to be downloaded.
echo You can download TwitchDownloaderCLI-X.XX.X-Windows-x64.zip from: https://github.com/lay295/TwitchDownloader/releases/
echo Once downloaded, place the TwitchDownloaderCLI executable inside the root folder of this program.
echo Press any key once this is done.
pause

echo.
echo.
echo AutoClipper needs ffmpeg to be downloaded.
echo You can download it from: https://www.gyan.dev/ffmpeg/builds/
echo Once downloaded, place the ffmpeg executable inside the root folder of this program, or change the path to executable in the .env file if already installed.
echo Press any key once this is done.
pause

echo.
echo.
echo Creating Python venv...
call python -m venv venv
echo Installing Python deps...
call "venv/Scripts/pip.exe" install -r requirements.txt
echo.
echo Finished installing, use RUN.bat to execute the program.
pause
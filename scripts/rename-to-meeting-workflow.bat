@echo off
REM Run after closing Cursor/terminals using this folder.
cd /d D:\Projects
if exist meeting-workflow (
  echo meeting-workflow already exists - aborting.
  exit /b 1
)
ren meeting-pipeline meeting-workflow
echo Renamed to D:\Projects\meeting-workflow
pause

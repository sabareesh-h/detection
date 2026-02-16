@echo off
echo Stopping CVAT...
cd cvat
docker compose down
echo CVAT stopped.
pause

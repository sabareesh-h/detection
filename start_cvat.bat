@echo off
echo Starting CVAT...
cd cvat
docker compose up -d
echo CVAT containers started.
echo Opening CVAT in browser...
start http://localhost:8080
pause

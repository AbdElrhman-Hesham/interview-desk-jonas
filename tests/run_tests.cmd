@echo off
echo === Python syntax checks ===
python -m py_compile app\interview.py && echo   interview.py OK || echo   interview.py FAIL
python -m py_compile app\interview_report.py && echo   interview_report.py OK || echo   interview_report.py FAIL
python -m py_compile app\server.py && echo   server.py OK || echo   server.py FAIL
python -m py_compile app\engine.py && echo   engine.py OK || echo   engine.py FAIL
python -m py_compile tests\test_interview_selection.py && echo   test_interview_selection.py OK || echo   test_interview_selection.py FAIL

echo.
echo === Unit tests ===
python -m unittest tests.test_interview_selection -v

echo.
echo === JavaScript syntax check ===
node -e "const fs=require('fs');const src=fs.readFileSync('app/static/index.html','utf8');const m=src.match(/<script>([\s\S]*?)<\/script>/);if(!m){process.exit(1)}try{new Function(m[1]);console.log('  index.html JS: OK')}catch(e){console.error('  index.html JS: FAIL -',e.message);process.exit(1)}"

echo.
pause

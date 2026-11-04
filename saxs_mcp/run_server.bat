@echo off
rem HEPS SAXS MCP server launcher (ASCII only: cmd parses this file as GBK,
rem  non-ASCII comments would corrupt line parsing and cause syntax errors)
rem Env root is configurable: set SAXSAGENT_ENV (or let mcp.json env block do
rem it); fallback keeps the developer-machine default for local use.
if "%SAXSAGENT_ENV%"=="" set "SAXSAGENT_ENV=E:\CondaEnvs\saxsagent"
rem Prepend the env dirs to PATH so LAPACK/OpenBLAS delay-loading resolves to
rem THIS env's DLLs, not a polluted parent env (crash 0xC06D007F).
set "PATH=%SAXSAGENT_ENV%\Library\bin;%SAXSAGENT_ENV%\DLLs;%SAXSAGENT_ENV%\Scripts;%SAXSAGENT_ENV%;%PATH%"
rem Force UTF-8: sasmodels reads C source files with default open(); without
rem this the MCP process uses GBK and crashes on ASCII files (UnicodeDecodeError).
set "PYTHONUTF8=1"
"%SAXSAGENT_ENV%\python.exe" "%~dp0server.py"

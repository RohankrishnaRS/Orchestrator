# Start the local portable PostgreSQL 16 (no Docker, no admin). Data: %USERPROFILE%\pg16\data
$PG = "$env:USERPROFILE\pg16\pgsql\bin"; $DATA = "$env:USERPROFILE\pg16\data"
if (-not (Test-Path "$PG\pg_ctl.exe")) { throw "Portable PostgreSQL not found at $PG - see docs/persistence.md" }
& "$PG\pg_ctl.exe" -D $DATA status *> $null
if ($LASTEXITCODE -ne 0) {
  & "$PG\pg_ctl.exe" -D $DATA -l "$env:USERPROFILE\pg16\pg.log" -o "-p 5432 -c listen_addresses=127.0.0.1" start
}
& "$PG\pg_isready.exe" -h 127.0.0.1 -p 5432
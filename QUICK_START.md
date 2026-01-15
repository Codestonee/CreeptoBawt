# SERQET Quick Start

## First Time (3-5 min)

```powershell
.\start.ps1 -Fresh
```

## Daily Start (5 sec)

```powershell
.\start.ps1
```

## Stop

```powershell
docker-compose down
```

## Restart (5 sec)

```powershell
.\restart.ps1
```

## View Logs

```powershell
docker-compose logs -f trading-bot
```

## Development Mode

```powershell
.\dev.ps1  # Code changes auto-reload
```

## Clean Rebuild (5 min)

```powershell
.\rebuild.ps1  # Use if dependencies changed
```

## Troubleshooting

### Slow builds?

- Check Docker is using WSL2: `docker info`
- Exclude from Windows Defender
- Increase Docker memory to 4GB

### Container won't start?

```powershell
docker-compose logs trading-bot
```

### Database locked?

```powershell
docker-compose down
Remove-Item data/*.db-shm, data/*.db-wal
docker-compose up -d
```

# BSM Dashboard - Quick Reference

## To Restart Server
```bash
export OPENAI_API_KEY="your-openai-api-key"
cd /home/Fate/.openclaw/workspace/bsm-exhibition
python3 server.py &
```

## To Create Tunnel
```bash
cloudflared tunnel --url http://localhost:5000
```

## Files
- `/home/Fate/.openclaw/workspace/bsm-exhibition/` - Main project
- `/home/Fate/.openclaw/workspace/memory/bsm/project.md` - Full documentation

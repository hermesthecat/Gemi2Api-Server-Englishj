# Gemi2Api-Server
A simple server-side implementation of [HanaokaYuzu / Gemini-API](https://github.com/HanaokaYuzu/Gemini-API)

[![pE79pPf.png](https://s21.ax1x.com/2025/04/28/pE79pPf.png)](https://imgse.com/i/pE79pPf)

## Quick Deployment

### Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/zhiyu1998/Gemi2Api-Server)

### HuggingFace (Deployed by @qqrr)

[![Deploy to HuggingFace](https://img.shields.io/badge/%E7%82%B9%E5%87%BB%E9%83%A8%E7%BD%B2-%F0%9F%A4%97-fff)](https://huggingface.co/spaces/ykl45/gmn2a)

## Direct Run

0. Fill in `SECURE_1PSID` and `SECURE_1PSIDTS` (find them in browser developer tools cookies after logging into Gemini), and optionally `API_KEY`.
```properties
SECURE_1PSID = "COOKIE VALUE HERE"
SECURE_1PSIDTS = "COOKIE VALUE HERE"
API_KEY= "API_KEY VALUE HERE"
```
1. Install dependencies with `uv`
> uv init
> 
> uv add fastapi uvicorn gemini-webapi

> [!NOTE]  
> If `pyproject.toml` exists, use the following command:  
> uv sync

Alternatively, `pip` can also be used:

> pip install fastapi uvicorn gemini-webapi

2. Activate the environment
> source venv/bin/activate

3. Start the server
> uvicorn main:app --reload --host 127.0.0.1 --port 8000

> [!WARNING] 
> tips: If API_KEY is not provided, it will be used directly.

## Run with Docker (Recommended)

### Quick Start

1. Clone this project
   ```bash
   git clone https://github.com/zhiyu1998/Gemi2Api-Server.git
   ```

2. Create a `.env` file and fill in your Gemini Cookie credentials:
   ```bash
   cp .env.example .env
   # Open the .env file with an editor and fill in your Cookie values
   ```

3. Start the service:
   ```bash
   docker-compose up -d
   ```

4. The service will be running on http://0.0.0.0:8000

### Other Docker Commands

```bash
# View logs
docker-compose logs

# Restart service
docker-compose restart

# Stop service
docker-compose down

# Rebuild and start
docker-compose up -d --build
```

## API Endpoints

- `GET /`: Service status check
- `GET /v1/models`: Get list of available models
- `POST /v1/chat/completions`: Chat with the model (similar to OpenAI interface)

## Common Issues

### Server 500 Error Solution

500 errors are generally due to IP issues or too frequent requests (for the latter, wait for a while or log in again with a new incognito tab to get new Secure_1PSID and Secure_1PSIDTS). See issues:
- [__Secure-1PSIDTS · Issue #6 · HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API/issues/6)
- [Failed to initialize client. SECURE_1PSIDTS could get expired frequently · Issue #72 · HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API/issues/72)

Solution steps:
1. Access [Google Gemini](https://gemini.google.com/) in an incognito tab and log in.
2. Open browser developer tools (F12).
3. Switch to the "Application" tab.
4. Find "Cookies" > "gemini.google.com" on the left.
5. Copy the values of `__Secure-1PSID` and `__Secure-1PSIDTS`.
6. Update the `.env` file.
7. Rebuild and start: `docker-compose up -d --build`

## Contributions

Thanks to the following developers for their contributions to `Gemi2Api-Server`:

<a href="https://github.com/zhiyu1998/Gemi2Api-Server/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zhiyu1998/Gemi2Api-Server&max=1000" />
</a>
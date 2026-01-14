# Titan HFT Bot 🚀

Hey there! Welcome to **Titan**, a high-frequency trading bot I built for crypto futures. It's designed to be fast, safe, and modular. I started this because I wanted something more robust than the standard "while True" loops you see in most tutorials.

This bot is fully async, uses a proper event architecture, and includes a real-time dashboard so you can actually see what's going on without staring at logs all day.

## What's Inside?

- **⚡ Async Core:** Built on `asyncio` and `uvloop` for low-latency execution.
- **🛡️ Risk First:** Has a dedicated `RiskGatekeeper` that validates every single order _before_ it goes to the exchange. It tracks position limits, daily loss, and "fat finger" errors.
- **🧠 Strategies:**
  - **Avellaneda-Stoikov:** A classic market-making strategy that quotes around the mid-price.
  - **Funding Arbitrage:** (In beta) For capturing funding rate discrepancies.
- **📊 Real-time Dashboard:** A Streamlit app that shows your PnL, active positions, and open orders live.
- **🐋 Shadow Order Book:** Maintains a local L2 order book for true mid-price calculation (crucial for accurate quoting).
- **ACID-compliant Order Manager:** Keeps local state perfectly synced with the exchange to prevent "ghost orders."

## Getting Started

### Prerequisites

You'll need **Python 3.11+**. I highly recommend using a virtual environment.

### Installation

1.  **Clone the repo:**

    ```bash
    git clone https://github.com/yourusername/CreeptBaws.git
    cd CreeptBaws
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up configuration:**
    Copy the example env file and add your keys.
    ```bash
    cp .env.example .env
    ```
    _Open `.env` and fill in your Binance API Key and Secret._

### Running the Bot

To start the trading engine (by default it runs on Binance Testnet):

```bash
python main.py
```

You should see logs starting up, connecting to WebSocket streams, and initializing the strategies.

### Launching the Dashboard

Want to see the pretty charts? Open a new terminal and run:

```bash
streamlit run dashboard/app.py
```

Then head to `http://localhost:8501` in your browser.

## Docker Support 🐳

If you prefer keeping things containerized (or want to run this on a VPS):

```bash
docker-compose up --build -d
```

This spins up the bot, the dashboard, and a Redis/Postgres instance if you have them configured.

## Risk Warning ⚠️

**Please read this:** Trading crypto futures involves significant risk. This bot is a tool, not a money printer.

- Start with **Paper Trading** (enabled by default in `config/settings.py`).
- Test thoroughly on Testnet before risking real funds.
- I am not responsible for any blown accounts!

## Contributing

Found a bug? Have an idea for a new strategy? Feel free to open an issue or submit a PR. I'm always looking to optimize the execution speed!

Happy Trading! 📈

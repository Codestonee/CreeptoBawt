"""Quick test for Telegram integration."""
import asyncio
from utils.telegram_alerts import get_telegram_alerter

async def main():
    alerter = get_telegram_alerter()
    print(f"Telegram enabled: {alerter.enabled}")
    
    if alerter.enabled:
        result = await alerter.send(
            "🤖 <b>Titan HFT</b>\n\n"
            "✅ Telegram integration working!\n\n"
            "You will receive alerts for:\n"
            "• Circuit breaker triggered\n"
            "• Emergency close executed\n"
            "• PnL threshold breach"
        )
        print(f"Message sent: {result}")
    else:
        print("Telegram not configured. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")

if __name__ == "__main__":
    asyncio.run(main())

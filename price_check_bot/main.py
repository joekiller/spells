# In main.py

import os
from dotenv import load_dotenv
from price_check_bot import PriceCheckBot
from models import S010Prompt, S011More

load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
channel_ids = [1121234097981898792, 1121650892719013988]

channel_models = {
    1121234097981898792: [S010Prompt(), S011More()],
}

[[model.load() for model in models] for models in channel_models.values()]


def main():
    bot = PriceCheckBot(TOKEN, channel_models)
    bot.run()


if __name__ == "__main__":
    main()

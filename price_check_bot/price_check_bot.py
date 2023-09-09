# In price_check_bot.py
from typing import Dict

import discord

from models import SplackModel
from version import __version__


class PriceCheckBot(discord.Client):
    def __init__(self, token, channel_models: Dict[int, list[SplackModel]]):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents)
        self.channel_models = channel_models
        self.token = token
        self.channel_ids = set([c for c in self.channel_models.keys()])

    async def on_ready(self):
        print(f'We have logged in as {self.user}')

    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.channel.id not in self.channel_ids:
            return

        if message.content.startswith('@pc:') and message.channel.id in self.channel_models:
            query = message.content[4:154].strip().lower()
            models = self.channel_models[message.channel.id]
            result = f'{query}: '
            footer = 'models: '
            results = []
            footers = []
            for model in models:
                prediction = model.predict(query)
                results.append(str(round(prediction[0][0], 1)))
                footers.append(model.version)
            result += ' to '.join(results)
            footer += ', '.join(footers)
            print(result, footer)
            await message.channel.send(result)

        if message.content == "@version" and message.channel.id in self.channel_models:
            version = __version__
            await message.channel.send(f"The version of joe-spell-prediction is {version}")

        if message.content == "@models" and message.channel.id in self.channel_models:
            models = self.channel_models[message.channel.id]
            await message.channel.send(f"Models: {', '.join([model.version for model in models])}")

    def run(self):
        super().run(self.token)

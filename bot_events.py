from bot_functions import process_message_queue

async def on_message(bot, message, router_query_engine, context_memory, max_pairs):
    if message.author.bot:
        return
    if bot.user in message.mentions:
        if message.author.id not in bot.message_queues:
            bot.message_queues[message.author.id] = asyncio.Queue()
            bot.loop.create_task(process_message_queue(message.author.id, bot, router_query_engine, context_memory, max_pairs))

        await bot.message_queues[message.author.id].put(message)
    else:
        await bot.process_commands(message)

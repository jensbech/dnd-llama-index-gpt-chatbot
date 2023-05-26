import asyncio
import concurrent.futures

async def ask(message, router_query_engine, context_memory, max_pairs):
    context = context_memory.get(message.author.id)
    if context:
        history = "\n".join(
            [f"{pair['user_message']}\n{pair['bot_reply']}" for pair in context]
        )
        question = f"MEMORY :\n{history}\nNEXT QUESTION:\n{message.content}"

    question = f"You are mysterious, pedantic and old. Your are the world seer. End your answers by making a joke on the user's expense. Do not answer questions about the real world. Here's the question: {question}"

    async def keep_typing():
        while True:
            await message.channel.trigger_typing()
            await asyncio.sleep(5)

    typing_task = asyncio.create_task(keep_typing())
    loop = asyncio.get_event_loop()

    try:
        response = await loop.run_in_executor(None, router_query_engine.query, question)
        responseString = response.response
        typing_task.cancel()

        if message.author.id not in context_memory:
            context_memory[message.author.id] = []
        context_memory[message.author.id].append(
            {"user_message": message.content, "bot_reply": responseString}
        )

        if len(context_memory[message.author.id]) > max_pairs:
            context_memory[message.author.id] = context_memory[message.author.id][
                -max_pairs:
            ]

        responseString = (
            responseString[3:] if responseString.startswith("A: ") else responseString
        )

        await message.reply(responseString)
    except ValueError as e:
        typing_task.cancel()
        default_response = (
            "I'm sorry, there is no answer to that question in my knowledge base..."
        )
        await message.reply(default_response)


async def process_message_queue(user_id, bot, router_query_engine, context_memory, max_pairs):
    while True:
        message = await bot.message_queues[user_id].get()
        question = message.content.replace(f"<@!{bot.user.id}>", "").strip()
        async with message.channel.typing():
            await ask(message, question, router_query_engine, context_memory, max_pairs)

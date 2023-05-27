# DnD Llama-Index GPT Discord Chatbot
Readme written by GPT4.

This repository hosts a Discord bot designed to serve as a custom knowledge database for your Dungeons and Dragons (D&D) campaign. The bot, leveraging the capabilities of OpenAI's language model, GPT-3.5-turbo, offers real-time interaction with the dataaset.

Example usage:
file:///home/jens-huawei/Downloads/Screenshot%20from%202023-05-27%2012-10-07(1).png![image](https://github.com/jensbech/dnd-llama-index-gpt-chatbot/assets/8881797/2124fc28-2b03-47ab-b362-f1d04ffa1068)

The chatbot utilizes llama-index, a feature-rich machine learning library that provides utilities for indexing and querying datasets. The bot leverages this functionality to create a queryable index from a Markdown-based wiki representing your DnD campaign universe.

## What Does This Code Do?

This bot constructs a custom knowledge graph based on the content of your D&D campaign's wiki, stored as Markdown files. The bot loads each document, creates a vector index using GPT-3.5-turbo, and then generates a comprehensive summary. This summary serves as the bot's understanding of the document content. All the vector indices are then composed into a graph to enable complex querying.

When a query is received from Discord, it's processed using a query engine that breaks down the question and routes it to the most relevant part of the knowledge graph. This ensures that the bot's response is tailored to the user's query, offering precise and context-aware responses.

Importantly, all this occurs asynchronously with Discord, allowing the bot to simulate typing in real-time. This maintains the interactive and immersive nature of the D&D campaign.

## Introduction

This bot aims to bridge the gap between static campaign wikis and real-time player interactions. Instead of players referring to the campaign wiki or the Game Master (GM) for context, this bot answers questions about the campaign universe dynamically, offering a depth of knowledge and context-aware humour that's expected from a GM.

For a better user experience, the bot has a specific personality: it's designed to be mysterious, pedacious, and old, like a world seer. Additionally, it ends each answer with a context-relevant joke or fun comment. This helps to elevate the bot beyond a simple query tool, enabling it to play a character role in the campaign itself.

## Getting Started

To get started with the bot, follow these steps:

1. Clone the repository
2. Install the necessary dependencies
	1. llama-index
	2. discord.py
	3. python.dotenv
	4. langchain.chat_models
	5. asyncio
3. Set up your `file_index.json` and `.md` files in the appropriate directories
4. Provide your `DISCORD_TOKEN` and `OPENAI_API_KEY` in a `.env` file
5. Run the bot
6. Host it on your Raspberry Pi or somewhere else.
7. ???
8. Profit

Note: Ensure your Markdown files are correctly formatted, and your `file_index.json` contains relevant metadata for each file.

For those who prefer a more skeptical take on the campaign universe, check out the separate branch featuring a bot with a "conspiracy theorist" personality.

from mas import AgentSystemManager

# Running a Telegram Bot using MAS can be done with just one line of code
# Make sure to have the necessary API keys defined in api_keys.json (or .env, if you prefer)
# Particularly, you'll need to create a bot token using Telegram's Bot Father
# Note that we are not specifying a component to run and the config.json does not include any explicit automation, so an implicit automation will be defined by the manager by running all components linearly in the order they are defined in the file.
AgentSystemManager("config.json").start_telegram_bot(verbose=True)
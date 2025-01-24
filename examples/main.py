from mas import AgentSystemManager

# Initialize the manager
manager = AgentSystemManager(config_json="config.json") 
output = manager.run(input="Hello world!", verbose=True)

print(output)

# print(manager.show_history()) # alternatively, you can show full message history
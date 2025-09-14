from mas import AgentSystemManager

# Initialize the manager
manager = AgentSystemManager(config="config.json") 
output = manager.run(input="Hello world!", verbose=True)

print(output)

# print(manager.show_history()) # alternatively, you can show full message history
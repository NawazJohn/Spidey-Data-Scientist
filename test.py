from agent.agent import AutoDSAgent

# 1. Initialize our new AI agent
agent = AutoDSAgent()

# 2. Give it a fake dataset profile
fake_profile = {
    "rows": 1000,
    "columns": 5,
    "missing_cells": 150,
    "target_column": "price",
    "column_types": {"price": "float", "age": "int", "category": "string"}
}

# 3. Observe the profile
agent.observe(fake_profile)

# 4. Ask the LLM to decide the next step
decision = agent.decide(fake_profile)
print("The AI decided to:", decision)

from agent.agent import AutoDSAgent

# Initialize the agent
agent = AutoDSAgent()

# Give it a completely different dataset profile
fake_profile_2 = {
    "rows": 10000,
    "columns": 20,
    "missing_cells": 0,  # No missing data this time!
    "target_column": "is_fraud",
    "target_distribution": {"0": 9900, "1": 100}, # Highly imbalanced!
    "column_types": {"is_fraud": "int", "amount": "float", "merchant": "string", "location": "string"}
}

# Observe the profile
agent.observe(fake_profile_2)

# Ask the LLM to decide the next step
decision = agent.decide(fake_profile_2)
print("=== Scenario: Fraud Detection (Imbalanced Data, No Missing Values) ===")
print("The AI decided to:", decision)

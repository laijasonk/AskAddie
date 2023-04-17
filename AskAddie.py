#!/usr/bin/env python3

from src.ChatBots import BloomBot
from src.ChatBots import DollyBot

# Initiate a chatbot named Addie with BLOOM
model_name = "bigscience/bloom-3b"
tokenizer_name = "bigscience/bloom-3b"
addie = BloomBot()

# # Initiate a chatbot named Addie with Dolly
# model_name = "databricks/dolly-v2-3b"
# tokenizer_name = "databricks/dolly-v2-3b"
# addie = DollyBot()

# Set and load the models
addie.set_model_names(model_name, tokenizer_name)
addie.load_models()

# Prepare prompt
addie.use_default_prompt()

# Run the conversational AI agent
addie.run()

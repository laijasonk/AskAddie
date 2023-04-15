#!/usr/bin/env python3

import os
import glob
from src.Agents import BloomBot

# Initiate a chatbot named Addie
addie = BloomBot()

# Set and load the models
model_name = "bigscience/bloom-1b1"
tokenizer_name = "bigscience/bloom-1b1"
addie.set_model_names(model_name, tokenizer_name)
addie.load_models()

# Prepare prompt
addie.use_default_prompt()

# Run the conversational AI agent
addie.run()

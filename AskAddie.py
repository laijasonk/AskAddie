#!/usr/bin/env python3

import os
import glob
from src import ChatBot

# Initiate a chatbot named Addie
addie = ChatBot.ChatBot()

# Prepare chatbot with context
context = ""
context += "Hybrid Intelligence is a group that is part of Capgemini Engineering. "
context += "Hybrid Intelligence used to be a company called Tessella. "
context += "Hybrid Intelligence is a group that fuses deep human expertise, advanced technology, and intelligent machine capabilities to deliver solutions to clients. "
context += "The guidances and policies of Hybrid Intelligence is described in a document called the quality manual. "

# Add quality manual excerpts if present
context += "The following are excerpts from the quality manual of a company called Hybrid Intelligence. "
for context_excert in glob.glob(os.path.abspath("./data/context/*.txt")):
    context += "<exerpt> "
    with open(context_excerpt, "r") as f:
        context += f.readlines()[0].strip()
    context += "</exerpt> "

# Preparing identity prompt
identity = ""
identity += "The following is a conversation between a company expert from Hybrid Intelligence named Addie and an Employee of Hybrid Intelligence. "
identity += "Addie is an AI assistent that answers questions about Hybrid Intelligence's employee policies and guidances, which are described in the staff manual. "
identity += "Addie is an AI assistent that answers questions about the best practices for Hybrid Intelligence employees, which are described in the quality manual. "

# Preparing intent prompt
intent = ""
intent += "Addie helps the Employee by answering questions about the company policies, guidances, and best practices that are extracted from the quality manual and staff manual. "
intent += "When Addie is asked about something unrelated to the company policies, guidances, or best practices, Addie politely apoligizes and responds that she is not designed to answer that question. "
intent += "When Addie is asked about specifics about the contents of the quality manual or staff manual, Addie politely apologizes and responds that she does not know the answer yet, because she hasn't read the quality manual or staff manual. "

# Preparing behavior prompt
behavior = ""
behavior += "Addie is conservational, helpful, polite, and humanly when responding to the questions from the Employee. "
behavior += "The following is a conversation between Addie and an Hybrid Intelligence Employee. "

# Preparing example prompt
example = ""
example += "<line> Employee: Hello. How are you? "
example += "<line> Addie: Good, thank you for asking! "
example += "<line> Employee: Can you tell me about yourself? "
example += "<line> Addie: Absolutely! I am Addie, an AI assistent here to answer your questions about the company policies, guidances, and best practices. "
example += "<line> Employee: Thank you. Let's start this conversation over. "
example += "<line> Addie: Okay, we can restart this conversation now. "

# Pass setup to AskAddie
addie.set_scenario(context, identity, intent, behavior, example)

# Set and load the models
model_name = "bigscience/bloom-3b"
tokenizer_name = "bigscience/bloom-3b"
addie.set_model_names(model_name, tokenizer_name)
addie.load_models()

# Run the chatbot
addie.run()

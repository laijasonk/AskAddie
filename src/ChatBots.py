import os
import glob

# Set before importing libraries to change the cache directory
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./data/models/")

# Model imports
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class ChatBot:
    """Base class to initialize and run AI conversational agents.

    Attributes:
        model_name (str): Name of the Hugging Face model
        tokenizer_name (str): Name of the Hugging Face model
        model (obj): Pre-trained model
        tokenizer (obj): Pre-trained tokenizer
        context (str): Background information to include into prompt
        identity (str): Prompt engineer for chatbot's identity
        intent (str): Prompt engineer for chatbot's intent
        behavior (str): Prompt engineer for chatbot's behavior
        example (str): Example conversation format
        key (str): Example new response line format
    """

    def __init__(self):
        """Default constructor.

        Args:
            None
        Returns:
            None
        """

        # Default values
        self.model_name = "bigscience/bloom-3b"
        self.tokenizer_name = "bigscience/bloom-3b"
        self.model = None
        self.tokenizer = None
        self.context = "Hybrid Intelligence is a practice unit inside a company called Capgemini Engineering.\n"
        self.identity = "The following is a conversation between Addie and a Hybrid Intelligence employee.\n"
        self.intent = "Addie is helping the Employee navigate the quality manual.\n"
        self.behavior = "Addie is conversational, helpful, and friendly.\n"
        self.example = "<line> Employee: How are you?\n<line> Addie: Good, thank you!\n"
        self.key = "<line> Addie:"

        return None

    def load_models(self, model_name=None, tokenizer_name=None):
        """Load the models (fetch if non-existent).

        Args:
            model_name (str): Name of the Hugging Face model
            tokenizer_name (str): Name of the Hugging Face model
        Returns:
            None
        """

        # Keep preset names if values are empty (override otherwise)
        if model_name == None:
            pass
        else:
            self.model_name = model_name
        if tokenizer_name == None:
            pass
        else:
            self.tokenizer_name = tokenizer_name

        print(f"Loading model ({self.model_name})")
        self.model = self._pretrained_model(self.model_name)

        print(f"Loading tokenizer ({self.tokenizer_name})")
        self.tokenizer = self._pretrained_tokenizer(self.tokenizer_name)

        return None

    def set_model_names(self, model_name, tokenizer_name):
        """Set the variable for model and tokenizer names.

        Args:
            model_name (str): Name of the Hugging Face model
            tokenizer_name (str): Name of the Hugging Face model
        Returns:
            None
        """

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name

        return None

    def set_scenario(self, context, identity, intent, behavior, example, key):
        """Set the scenario via prompt engineering.

        Args:
            context (str): Background information to include into prompt
            identity (str): Prompt engineer for Addie's identity
            intent (str): Prompt engineer for Addie's intent
            behavior (str): Prompt engineer for Addie's behavior
            example (str): Example conversation format
            key (str): Example new response line format
        Returns:
            None
        """

        self.context = context
        self.identity = identity
        self.intent = intent
        self.behavior = behavior
        self.example = example
        self.key = key

        return None

    def run(self):
        """Run the AI conversational agent.

        Args:
            None
        Returns:
            None
        """

        # Create an initial prompt
        prompt = (
            self.context + self.identity + self.intent + self.behavior + self.example
        )
        count = len(prompt.split("<line>"))

        # Some arbitrary text (does not go into prompt)
        print()
        print("Starting AI conversational agent")
        print("Addie: Hello, my name is Addie. How can I help you?")
        print()

        # Run the chatbot
        while True:
            user_input = input("Input: ")
            formatted_input = f"<line> Employee: {user_input.strip()}\n"
            prompt += formatted_input
            prompt += self.key
            count = len(prompt.split("<line>"))

            # print(prompt) # debug
            prompt = self._ai_response(prompt)
            # print(prompt) # debug
            [prompt, last_line] = self._parse_prompt(prompt, count)
            # print(prompt) # debug

            print(last_line)
            print()

        return None

    def use_default_prompt(self):
        """Set the class object to use default prompts.

        Args:
            None
        Returns:
            None
        """

        pass

    def _pretrained_model(self, model_name):
        """Private method to load pretrained model.

        Args:
            model_name (str): Name of the Hugging Face model
        Returns:
            model (obj): Pre-trained model
        """

        return AutoModelForCausalLM.from_pretrained(model_name)

    def _pretrained_tokenizer(self, tokenizer_name):
        """Private method to load pretrained tokenizer.

        Args:
            tokenizer_name (str): Name of the Hugging Face model
        Returns:
            tokenizer (obj): Pre-trained tokenizer
        """

        return AutoTokenizer.from_pretrained(tokenizer_name)

    def _ai_response(self, prompt):
        """Generate text based on initial prompt.

        Args:
            prompt (str): Initial prompt to append generated text
        Returns:
            updated_prompt (str): Generated text appended to the initial prompt
        """

        # Text clean-up
        prompt = prompt.replace("  ", " ")

        # Tokenize input prompt with Hugging Face tools
        inputs = self.tokenizer(prompt, return_tensors="pt")
        attention_mask = inputs.get("attention_mask", None)
        pad_token_id = self.tokenizer.pad_token_id

        # Generate text on top of prompt
        updated_prompt = self.tokenizer.decode(
            self.model.generate(
                inputs["input_ids"],
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                # num_beams=2,
                # no_repeat_ngram_size=2,
                # do_sample=True,
                # top_k=50,
                # top_p=0.92,
                max_new_tokens=200,
                early_stopping=True,
            )[0]
        )

        return updated_prompt

    def _parse_prompt(self, prompt, count):
        """Format the generated text.

        Args:
            prompt (str): Initial prompt to generate text on top of.
            count (int): Index where the last response was provided
        Returns:
            updated_prompt (str): Updated prompt after cleaning
            response (str): The chatbot repsonse
        """

        split_prompt = prompt.split("<line>")
        updated_prompt = "<line>".join(split_prompt[0:count])
        updated_prompt = updated_prompt.replace("  ", " ")
        updated_prompt = updated_prompt.strip() + "\n"

        response = split_prompt[count - 1]
        response = response.strip()

        return [updated_prompt, response]


class BloomBot(ChatBot):
    def use_default_prompt(self):
        # Prepare chatbot with context
        context = "Hybrid Intelligence is a practice unit inside a company called Capgemini Engineering. "
        context += "Hybrid Intelligence used to be a UK-based company called Tessella. "
        context += "The staff in Hybrid Intelligence fuses deep human expertise, advanced technology, and intelligent machine capabilities to deliver solutions to clients. "
        context += "The guidances and policies of Hybrid Intelligence is described in a document called the quality manual. "
        context += "\n"

        # Add quality manual excerpts if present
        context += (
            "The following are excerpts from Hybrid Intelligence's quality manual.\n"
        )
        for context_excerpt in glob.glob(os.path.abspath("./data/context/*.txt")):
            context += "<exerpt> "
            with open(context_excerpt, "r") as f:
                context += f.readlines()[0].strip()
            context += "\n"

        # Preparing identity prompt
        identity = "An AI assistent representing Hybrid Intelligence named Addie and an Employee of Hybrid Intelligence are having a conversation. "
        identity += "Addie is an AI assistent that answers questions about Hybrid Intelligence's employee policies and guidances, which are described in the staff manual. "
        identity += "Addie is an AI assistent that answers questions about the best practices for Hybrid Intelligence employees, which are described in the quality manual. "
        identity += "\n"

        # Preparing intent prompt
        intent = "Addie helps the Employee by answering questions about the company policies, guidances, and best practices that are extracted from the quality manual and staff manual. "
        intent += "\n"

        # Preparing behavior prompt
        behavior = "Addie is conservational, helpful, polite, and humanly when responding to the questions from the Employee."
        behavior += "\n"

        # Preparing example prompt
        example = "The following is a conversation between Addie and an Hybrid Intelligence Employee.\n"
        example += "<line> Addie: Hello, I am an AI assistent named Addie. How can I help you?\n"
        example += "<line> Employee: Hello. How are you?\n"
        example += "<line> Addie: Good, thank you for asking!\n"
        example += "<line> Employee: Can you tell me about yourself?\n"
        example += "<line> Addie: Absolutely! I am Addie, an AI assistent here to answer your questions about Hybrid Intelligence's company policies, guidances, and best practices.\n"
        example += "<line> Employee: Thank you. Let's start this conversation over.\n"
        example += "<line> Addie: Hello, I am an AI assistent named Addie. How can I help you?\n"

        # Preparing start key prompt
        key = "<line> Addie:"

        # Set defaults
        self.set_scenario(context, identity, intent, behavior, example, key)

        return None


class DollyBot(ChatBot):
    def use_default_prompt(self):
        # Prepare chatbot with context
        context = "Complete the next line in the conversation.\n"
        context += "Hybrid Intelligence is a practice unit inside a company called Capgemini Engineering. "
        context += "Hybrid Intelligence used to be a UK-based company called Tessella. "
        context += "The staff in Hybrid Intelligence fuses deep human expertise, advanced technology, and intelligent machine capabilities to deliver solutions to clients. "
        context += "The guidances and policies of Hybrid Intelligence is described in a document called the quality manual. "
        context += "\n"

        # Add quality manual excerpts if present
        context += (
            "The following are excerpts from Hybrid Intelligence's quality manual.\n"
        )
        for context_excerpt in glob.glob(os.path.abspath("./data/context/*.txt")):
            context += "<exerpt> "
            with open(context_excerpt, "r") as f:
                context += f.readlines()[0].strip()
            context += "\n"

        # Preparing identity prompt
        identity = "An AI assistent representing Hybrid Intelligence named Addie and an Employee of Hybrid Intelligence are having a conversation. "
        identity += "Addie is an AI assistent that answers questions about Hybrid Intelligence's employee policies and guidances, which are described in the staff manual. "
        identity += "Addie is an AI assistent that answers questions about the best practices for Hybrid Intelligence employees, which are described in the quality manual. "
        identity += "\n"

        # Preparing intent prompt
        intent = "Addie helps the Employee by answering questions about the company policies, guidances, and best practices that are extracted from the quality manual and staff manual. "
        intent += "\n"

        # Preparing behavior prompt
        behavior = "Addie is conservational, helpful, polite, and humanly when responding to the questions from the Employee. "
        behavior += "\n"

        # Preparing example prompt
        example = "The following is a conversation between Addie and an Hybrid Intelligence Employee.\n"
        example += "Complete the next line in the conversation below.\n"
        example += "<line> Addie: Hello, I am an AI assistent named Addie. How can I help you?\n"
        example += "<line> Employee: Hello. How are you?\n"
        example += "<line> Addie: Good, thank you for asking!\n"
        example += "<line> Employee: Can you tell me about yourself?\n"
        example += "<line> Addie: Absolutely! I am Addie, an AI assistent here to answer your questions about Hybrid Intelligence's company policies, guidances, and best practices.\n"
        example += "<line> Employee: Thank you. Let's start this conversation over.\n"
        example += "<line> Addie: Hello, I am an AI assistent named Addie. How can I help you?\n"

        # Preparing start key prompt
        key = "<line> Addie:"

        # Set defaults
        self.set_scenario(context, identity, intent, behavior, example, key)

        return None

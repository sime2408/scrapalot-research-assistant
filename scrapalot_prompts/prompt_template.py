from abc import ABC, abstractmethod
from langchain.prompts import load_prompt


class AbstractPromptTemplate(ABC):
    def __init__(self, filename):
        self.prompt_template = load_prompt(filename)

    @abstractmethod
    def prepare_prompt(self):
        pass


class ScrapalotSystemPromptTemplate(AbstractPromptTemplate):
    def prepare_prompt(self):
        # Prepare the prompt specific to ScrapalotSystemPromptTemplate
        return self.prompt_template


class ScrapalotSummaryPromptTemplate(AbstractPromptTemplate):
    def prepare_prompt(self):
        # Prepare the prompt specific to ScrapalotSummaryPromptTemplate
        return self.prompt_template

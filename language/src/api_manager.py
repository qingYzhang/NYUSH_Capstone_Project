from langchain.chains import LLMChain
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

import settings as s


def llm_chain(template: str, template_format: str = "f-string", temperature: float = 0.2,
              deployment_name: str = "GPT4") -> LLMChain:
    """Builds a langchain LLM Chain object."""
    optional_params = {"top_p": 0.95, "frequency_penalty": 0, "presence_penalty": 0}
    llm = AzureChatOpenAI(
        openai_api_type="azure",
        azure_endpoint="https://genai-api.prod1.nyumc.org/",
        openai_api_version="2024-02-01",
        deployment_name=deployment_name,
        openai_api_key=s.OPENAI_API_KEY_GPT4,
        request_timeout=120,
        max_retries=5,
        temperature=temperature,
        # model_kwargs=optional_params,
        max_tokens=2048,
        
        top_p=optional_params["top_p"],
        frequency_penalty=optional_params["frequency_penalty"],
        presence_penalty=optional_params["presence_penalty"],
    )

    human_message_prompt = HumanMessagePromptTemplate.from_template(template, template_format=template_format)
    prompt = ChatPromptTemplate.from_messages([human_message_prompt])

    return LLMChain(llm=llm, prompt=prompt)
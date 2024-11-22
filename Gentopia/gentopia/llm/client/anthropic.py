import os
from typing import List, Dict, Callable, Generator, Optional
import traceback

import anthropic

from gentopia.llm.base_llm import BaseLLM
from gentopia.model.agent_model import AgentOutput
from gentopia.model.completion_model import *
from gentopia.model.param_model import *
import json

class AnthropicClaudeClient(BaseLLM, BaseModel):
    """
    Wrapper class for Anthropic Claude API.

    :param model_name: The name of the model to use (e.g. "claude-3-opus-20240229")
    :type model_name: str
    :param params: The parameters for the model
    :type params: AnthropicParamModel
    """
    model_name: str
    params: AnthropicParamModel = AnthropicParamModel()
    client: Optional[anthropic.Anthropic] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_param(self) -> AnthropicParamModel:
        return self.params

    def completion(self, prompt: str, **kwargs) -> BaseCompletion:
        """
        Completion method for Anthropic Claude API.

        :param prompt: The prompt to use for completion
        :type prompt: str
        :param kwargs: Additional keyword arguments
        :return: BaseCompletion object
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.params.max_tokens,
                temperature=self.params.temperature,
                system=self.params.system,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return BaseCompletion(
                state="success",
                content=response.content[0].text,
                prompt_token=response.usage.input_tokens,
                completion_token=response.usage.output_tokens
            )
        except Exception as exception:
            print("Exception:", exception)
            return BaseCompletion(state="error", content=str(exception))

    def chat_completion(self, messages: List[dict]) -> ChatCompletion:
        """
        Chat completion method for Anthropic Claude API.

        :param messages: List of message dictionaries with role and content
        :type messages: List[dict]
        :return: ChatCompletion object
        """
        try:
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    self.params.system = msg["content"]
                    continue
                anthropic_messages.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                })

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.params.max_tokens,
                temperature=self.params.temperature,
                system=self.params.system,
                messages=anthropic_messages
            )

            return ChatCompletion(
                state="success",
                role="assistant",
                content=response.content[0].text,
                prompt_token=response.usage.input_tokens,
                completion_token=response.usage.output_tokens
            )
        except Exception as exception:
            print("Exception:", exception)
            return ChatCompletion(state="error", content=str(exception))

    def stream_chat_completion(self, messages: List[dict], **kwargs) -> Generator:
        """
        Stream chat completion method for Anthropic Claude API.

        :param messages: List of message dictionaries with role and content
        :type messages: List[dict]
        :param kwargs: Additional keyword arguments
        :return: Generator yielding ChatCompletion objects
        """
        try:
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    self.params.system = msg["content"]
                    continue
                anthropic_messages.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                })

            stream = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.params.max_tokens,
                temperature=self.params.temperature,
                system=self.params.system,
                messages=anthropic_messages,
                stream=True,
                **kwargs
            )

            for response in stream:
                yield ChatCompletion(
                    state="success",
                    role="assistant",
                    content=response.delta.text if response.delta.text else "",
                    prompt_token=0,  # Tokens not available in stream mode
                    completion_token=0
                )

        except Exception as exception:
            print("Exception:", exception)
            yield ChatCompletion(state="error", content=str(exception))

    def function_chat_completion(self, message: List[dict],
                               function_map: Dict[str, Callable],
                               function_schema: List[Dict]) -> ChatCompletionWithHistory:
        """
        Function calling is not directly supported by Claude in the same way as OpenAI.
        This method provides a basic implementation that formats function schemas into the system prompt.

        :param message: List of message dictionaries with role and content
        :param function_map: Dictionary mapping function names to callable functions
        :param function_schema: List of function schemas
        :return: ChatCompletionWithHistory object
        """

        assert len(function_schema) == len(function_map)
        try:
            response = self.chat_completion(message)
            # print(response)

            try:
                # Try to parse response as function call
                function_call = json.loads(response.content)
                if "function" in function_call and "arguments" in function_call:
                    function_name = function_call["function"]
                    function_to_call = function_map[function_name]
                    function_response = function_to_call(**function_call["arguments"])

                    # Process function response
                    plugin_cost = 0
                    plugin_token = 0
                    if isinstance(function_response, AgentOutput):
                        plugin_cost = function_response.cost
                        plugin_token = function_response.token_usage
                        function_response = function_response.output
                    elif not isinstance(function_response, str):
                        raise Exception("Invalid tool response type. Must be one of [AgentOutput, str]")

                    # Add function response to message history
                    message.append({
                        "role": "assistant",
                        "content": json.dumps(function_call)
                    })
                    message.append({
                        "role": "function",
                        "name": function_name,
                        "content": str(function_response)
                    })

                    # Get final response
                    final_response = self.chat_completion(message)
                    message.append({
                        "role": "assistant",
                        "content": final_response.content
                    })

                    return ChatCompletionWithHistory(
                        state="success",
                        role="assistant",
                        content=final_response.content,
                        prompt_token=response.prompt_token + final_response.prompt_token,
                        completion_token=response.completion_token + final_response.completion_token,
                        message_scratchpad=message,
                        plugin_cost=plugin_cost,
                        plugin_token=plugin_token
                    )

            except json.JSONDecodeError:
                # If response is not JSON, treat as regular response
                message.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                return ChatCompletionWithHistory(
                    state="success",
                    role="assistant", 
                    content=response.content,
                    prompt_token=response.prompt_token,
                    completion_token=response.completion_token,
                    message_scratchpad=message
                )

        except Exception as exception:
            print("Exception:", exception)
            return ChatCompletionWithHistory(state="error", content=str(exception))
    

    def function_chat_stream_completion(self, messages: List[dict],
                                  function_map: Dict[str, Callable],
                                  function_schema: List[Dict]) -> Generator:
        """
        Stream function chat completion method for Anthropic Claude API.

        :param messages: List of message dictionaries with role and content
        :type messages: List[dict]
        :param function_map: Dictionary mapping function names to callable functions
        :param function_schema: List[Dict]
        :return: Generator yielding tuple of (type, ChatCompletionWithHistory)
        """
        try:
            # Process messages for Claude format
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    self.params.system = msg["content"]
                    continue
                anthropic_messages.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                })

            # Create streaming response
            stream = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.params.max_tokens,
                temperature=self.params.temperature,
                system=self.params.system,
                messages=anthropic_messages,
                stream=True
            )

            accumulated_text = ""
            for response in stream:
                # print(response)
                if hasattr(response, 'type'):
                    if response.type == 'content_block_delta':
                        delta_text = response.delta.text
                        accumulated_text += delta_text

                        try:
                            # Try to parse as JSON to check if it's a function call
                            parsed_json = json.loads(accumulated_text)
                            if isinstance(parsed_json, dict) and "function" in parsed_json and "arguments" in parsed_json:
                                # It's a complete function call
                                yield "function_call", ChatCompletionWithHistory(
                                    state="success",
                                    role="assistant",
                                    content=accumulated_text,
                                    function_call={
                                        "name": parsed_json["function"],
                                        "arguments": parsed_json["arguments"]
                                    },
                                    message_scratchpad=messages
                                )
                                return
                            else:
                                # Still accumulating potential JSON
                                continue
                        except json.JSONDecodeError:
                            # Not JSON, treat as regular response
                            yield "content", ChatCompletionWithHistory(
                                state="success",
                                role="assistant",
                                content=delta_text,
                                message_scratchpad=messages
                            )

        except Exception as exception:
            print("Exception in function_chat_stream_completion:", exception)
            yield "error", ChatCompletionWithHistory(
                state="error",
                content=str(exception),
                message_scratchpad=messages
            )
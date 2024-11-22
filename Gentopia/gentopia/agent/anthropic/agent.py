from typing import List, Union, Optional, Dict, Any, Type
import traceback

from gentopia import PromptTemplate
from gentopia.agent.base_agent import BaseAgent
from gentopia.llm.base_llm import BaseLLM
from gentopia.llm.client.anthropic import AnthropicClaudeClient
from gentopia.model.agent_model import AgentType, AgentOutput
from gentopia.output.base_output import BaseOutput
from gentopia.prompt import VanillaPrompt
from gentopia.memory.api import MemoryWrapper
from gentopia.utils.cost_helpers import calculate_cost


class AnthropicClaudeAgent(BaseAgent):
    """
    AnthropicClaudeAgent class with integrated memory support.
    
    :param name: Name of the agent
    :type name: str
    :param type: Type of the agent
    :type type: AgentType
    :param version: Version of the agent
    :type version: str
    :param description: Description of the agent
    :type description: str
    :param target_tasks: List of target tasks for the agent
    :type target_tasks: List[str]
    :param llm: Language model instance
    :type llm: AnthropicClaudeClient
    :param prompt_template: Template for prompts
    :type prompt_template: PromptTemplate
    :param plugins: List of available plugins
    :type plugins: List[Any]
    :param memory: Memory wrapper instance
    :type memory: Optional[MemoryWrapper]
    """
    name: str = "ClaudeAgent"
    type: AgentType = AgentType.anthropic
    version: str = "1.0"
    description: str = "Anthropic Claude Agent with function calling and memory capabilities"
    target_tasks: List[str] = []
    llm: AnthropicClaudeClient
    prompt_template: PromptTemplate = VanillaPrompt
    plugins: List[Any] = []
    memory: Optional[MemoryWrapper] = None
    message_scratchpad: List[Dict] = [{"role": "system", "content": "You are a helpful AI assistant."}]

    def __init__(self, **data):
        super().__init__(**data)
        self.initialize_system_message(f"Your name is {self.name}. You are described as: {self.description}")

    def initialize_system_message(self, msg: str):
        """Initialize the system message for the Claude agent.

        :param msg: System message to be initialized
        :type msg: str
        :raises ValueError: Raised if system message is modified after run
        """
        if len(self.message_scratchpad) > 1:
            raise ValueError("System message must be initialized before first agent run")
        self.message_scratchpad[0]["content"] = msg

    def run(self, instruction: str, output: Optional[BaseOutput] = None) -> AgentOutput:
        """Run the agent with the given instruction.

        :param instruction: Instruction to be run
        :type instruction: str
        :param output: Output manager object
        :type output: Optional[BaseOutput]
        :return: Agent output containing response and metrics
        :rtype: AgentOutput
        """
        self.clear()
        if output is None:
            output = BaseOutput()

        # Handle memory initialization if available
        if self.memory:
            self.memory.clear_memory_II()
            context_messages = self.memory.lastest_context(instruction, output)
            message_scratchpad = [self.message_scratchpad[0]] + context_messages
        else:
            message_scratchpad = self.message_scratchpad + [{"role": "user", "content": instruction}]

        total_cost = 0
        total_token = 0

        function_map = self._format_function_map()
        function_schema = self._format_function_schema()

        output.thinking(self.name)
        response = self.llm.function_chat_completion(message_scratchpad, function_map, function_schema)
        output.done()
        
        if response.state == "success":
            output.done(self.name)
            output.panel_print(response.content)
            
            # Handle memory updates if available
            if self.memory:
                # Save the main conversation to memory
                self.memory.save_memory_I(
                    {"role": "user", "content": instruction},
                    response.message_scratchpad[-1],
                    output
                )
                
                # If there was a function call, save it to secondary memory
                if len(response.message_scratchpad) > len(message_scratchpad) + 1:
                    self.memory.save_memory_II(
                        response.message_scratchpad[-3],  # Function call
                        response.message_scratchpad[-2],  # Function response
                        output,
                        self.llm
                    )
            else:
                self.message_scratchpad = response.message_scratchpad
            
            # Calculate costs and tokens
            total_cost += calculate_cost(self.llm.model_name, response.prompt_token,
                                        response.completion_token) + response.plugin_cost
            total_token += response.prompt_token + response.completion_token + response.plugin_token
            
            return AgentOutput(
                output=response.content,
                cost=total_cost,
                token_usage=total_token,
            )
        return AgentOutput(output=str(response.content), cost=0, token_usage=0)

    def stream(self, instruction: Optional[str] = None, output: Optional[BaseOutput] = None, is_start: bool = True) -> AgentOutput:
        """Stream output from the agent.

        :param instruction: Instruction to be run
        :type instruction: Optional[str]
        :param output: Output manager object
        :type output: Optional[BaseOutput]
        :param is_start: Whether this is the start of a conversation
        :type is_start: bool
        :return: Agent output containing response and metrics
        :rtype: AgentOutput
        """
        if output is None:
            output = BaseOutput()

        output.thinking(self.name)

        # Handle memory initialization if available
        if self.memory:
            if is_start:
                self.memory.clear_memory_II()
            context_messages = self.memory.lastest_context(instruction, output)
            message_scratchpad = [self.message_scratchpad[0]] + context_messages
        else:
            if instruction is not None:
                self.message_scratchpad.append({"role": "user", "content": instruction})
            message_scratchpad = self.message_scratchpad

        function_map = self._format_function_map()
        function_schema = self._format_function_schema()
        
        ans = []
        current_type = ''
        current_role = ''
        total_tokens = 0

        # print(self.llm.model_name) # going crazy
        
        try:
            for content_type, item in self.llm.function_chat_stream_completion(
                message_scratchpad, function_map, function_schema):
                
                if current_type == '':
                    output.done()
                    output.print(f"[blue]{self.name}: ")
                
                current_type = content_type
                current_role = item.role
                
                if item.state == "success":
                    ans.append(item.content)
                    total_tokens += len(item.content.split())
                    output.panel_print(item.content, f"[green]Response of [blue]{self.name}: ", True)
                
            result = ''.join(ans)
            output.clear()
            
            if current_type == "function_call":
                # Handle function call
                function_name = item.function_call["name"]
                function_to_call = function_map[function_name]
                
                output.update_status(f"Calling function: {function_name} ...")
                function_response = function_to_call(**item.function_call["arguments"])
                output.done()
                
                if isinstance(function_response, AgentOutput):
                    function_response = function_response.output
                
                output.panel_print(
                    function_response, 
                    f"[green]Function Response of [blue]{function_name}: "
                )
                
                # Save function call to memory if available
                function_call_msg = {
                    "role": "assistant",
                    "function_call": {
                        k: str(v) for k, v in item.function_call.items()
                    }
                }
                function_response_msg = {
                    "role": "function",
                    "name": function_name,
                    "content": str(function_response)
                }
                
                if self.memory:
                    self.memory.save_memory_II(
                        function_call_msg,
                        function_response_msg,
                        output,
                        self.llm
                    )
                else:
                    self.message_scratchpad.extend([
                        function_call_msg,
                        function_response_msg
                    ])
                
                # Continue the conversation
                return self.stream(output=output, is_start=False)
            else:
                # Save final response to memory if available
                if self.memory and instruction is not None:
                    self.memory.save_memory_I(
                        {"role": "user", "content": instruction},
                        {"role": current_role, "content": result},
                        output
                    )
            
            return AgentOutput(
                output=result,
                cost=calculate_cost(self.llm.model_name, total_tokens, total_tokens),
                token_usage=total_tokens
            )
            
        except Exception as e:
            output.panel_print(f"Error in stream: {str(e)}\n\n{traceback.format_exc()}")

            return AgentOutput(
                output=f"Error in stream: {str(e)}",
                cost=0,
                token_usage=0
            )

    def _format_function_schema(self) -> List[Dict]:
        """Format function schema for all plugins.

        :return: List of function schemas
        :rtype: List[Dict]
        """
        function_schema = []
        for plugin in self.plugins:
            if hasattr(plugin, 'args_schema'):
                parameters = plugin.args_schema.schema()
            else:
                parameters = {
                    "properties": {
                        "__arg1": {"title": "__arg1", "type": "string"},
                    },
                    "required": ["__arg1"],
                    "type": "object",
                }

            function_schema.append({
                "name": plugin.name,
                "description": plugin.description,
                "parameters": parameters,
            })
        return function_schema

    def clear(self):
        """Clear the message scratchpad except for the system message."""
        if self.memory:
            self.memory.clear_memory_II()
        self.message_scratchpad = self.message_scratchpad[:1]
from dataclasses import dataclass, field

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig
from searchagent.sources.base import SourceConfig

@dataclass
class AgentConfig():
    model: str = field(default="default")
    generation: GenerationHyperparameters = field(default_factory=GenerationHyperparameters)
    thinking: bool = field(default=True)
    max_tokens: int = field(default=32768)
    max_tokens_prompt_margin: int = field(default=5000)
    max_turn: int = field(default=100000)
    raise_repeat_tool_call: bool = field(default=True)
    system_prompt: str = field(
        default="""You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up on using more tools until you find the corresponding information. 
To get more information, make more tool calls with different arguments.
You should avoid producing repeated tool calls with identical arguments.
           
As you proceed, adhere to the following principles:

1. **Persistent Actions for Answers**: You will engage in many interactions of brief thinking and clear tool calls, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.

2. **Sufficient Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered through tool calls to confirm its accuracy and reliability.

3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.\n\nProvide your succinct final answer or state that you cannot find answer in a few words in <answer></answer> tags.
"""
    )
    question_prompt: str = field(
        default="""
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform brief reasoning and use the tool step by step, in an interleaved manner. You may call search multiple rounds
For example:
<think> your think </think>
<tool_call> 1 tool call per round </tool_call>
<tool_response> response from tool </tool_response>
<think> another round of think </think>
<tool_call> another round of tool calls</tool_call>
<tool_response> response from tool </tool_response>
...

Question: {Question}

Your response should be in the following format:
<think>{{your final thought on what should be the final answer}}</think>
<answer>{{your succinct, final answer, in a few words}}</answer>
""".strip()
    )
    max_tokens_prompt: str = field(
        default="You have now reached the maximum tool call turns you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:\n<answer>your answer</answer>"
    )
    max_turn_prompt: str = field(
        default="You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:\n<answer>your answer</answer>"
    )
    source: SourceConfig = field(default_factory=SourceConfig())


@dataclass
class WorkFlowConfig():
    training: bool = field(default=True)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: str = field(
        default="f1",
        metadata={
            "choices": ["f1", "llm_as_judge"]
        }
    )
    overlong_penalty_margin: int = field(default=5000)

@dataclass
class SearchAgentTrainingConfig(GRPOConfig):
    workflow: WorkFlowConfig = field(default_factory=WorkFlowConfig)
    eval_workflow: WorkFlowConfig = field(default_factory=WorkFlowConfig)
    dynamic_filter_fn: str | None = field(
        default="rollout_webagent.should_accept"
    )

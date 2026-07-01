"""Prompts and tool schemas used by the OpenSeeker BCP SFT eval path."""

from __future__ import annotations

from typing import Any


DEVELOPER_PROMPT = """You are a tool-augmented QA agent. Cleverly leverage appropriate tools to answer the user's question.

# Tools

You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "search", "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string"}, "description": "Array of query strings. Include multiple complementary search queries in a single call."}}, "required": ["query"]}}
{"name": "visit", "description": "Parse webpage(s) and return the summary of the content according to the goal.", "parameters": {"type": "object", "properties": {"url": {"type": ["string", "array"], "items": {"type": "string"}, "minItems": 1, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The goal of the visit for webpage(s)."}}, "required": ["url", "goal"]}}
</tools>

 If you decide to call tools, you MUST strictly follow the format below.

All tool calls must be wrapped inside <tool_calls_begin> and </tool_calls_end>.
Inside this block, each individual tool call must be wrapped with <tool_call> and </tool_call>.

The exact required format is:

<tool_calls_begin>
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
</tool_calls_end>

"""


SEARCH_DESCRIPTION = (
    "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
)

VISIT_DESCRIPTION = "Parse webpage(s) and return the summary of the content according to the goal."


TOOLS_VISIT: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": SEARCH_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of query strings. Include multiple complementary search queries in a single call.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": VISIT_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
                    },
                    "goal": {"type": "string", "description": "The goal of the visit for webpage(s)."},
                },
                "required": ["url", "goal"],
            },
        },
    },
]


VISIT_SUMMARY_PROMPT = """Please process the following webpage content and user goal to extract relevant information.

## Task Guidelines
1. Locate the portions that directly support the goal.
2. Extract the most relevant evidence.
3. Provide a concise summary and judge usefulness.
4. Find and provide (if any) urls to other pages that may be relevant.

Respond strictly in JSON:
{{
  "rational": "...",
  "evidence": "...",
  "summary": "...",
}}

## User Goal
{goal}

## Webpage Content
{webpage_content}
"""


JUDGE_PROMPT_BC_EN = """
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
- For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.

Below is a new question example. Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Evaluate this new question's predicted answer as one of the following:
A. [CORRECT]
B. [INCORRECT]

Return only the option representing [CORRECT] or [INCORRECT], i.e., just return A or B, without adding any other text.
""".strip()


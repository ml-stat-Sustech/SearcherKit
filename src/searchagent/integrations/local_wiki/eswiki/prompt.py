# 将每种模型的文档处理逻辑封装在自己的函数�?

def apply_e5_prompt(text: str) -> str:
    """�?E5 模型应用 'passage: ' 前缀"""
    return "passage: " + text


def apply_default_prompt(text: str) -> str:
    """默认策略，不进行任何修改"""
    return text


# 使用一个字典将策略名称映射到对应的函数
PROMPT_STRATEGIES = {
    "e5": apply_e5_prompt,
    "none": apply_default_prompt,
}

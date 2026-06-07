from __future__ import annotations


def add_searchagent_slime_arguments(parser):
    parser.add_argument(
        "--searchagent-agent-config",
        type=str,
        default=None,
        help="YAML file containing the SearchAgent agent config used by slime rollouts.",
    )
    parser.add_argument(
        "--searchagent-agent-config-key",
        type=str,
        default="agent",
        help="Dot path to the training AgentConfig inside --searchagent-agent-config.",
    )
    parser.add_argument(
        "--searchagent-eval-agent-config-key",
        type=str,
        default=None,
        help="Dot path to the eval AgentConfig; defaults to --searchagent-agent-config-key.",
    )
    parser.add_argument(
        "--searchagent-tool-call-parser",
        type=str,
        default="qwen",
        help="SGLang tool-call parser name used when the SearchAgent parser uses provider tools.",
    )
    parser.add_argument(
        "--searchagent-reasoning-parser",
        type=str,
        default="qwen3",
        help="SGLang reasoning parser name used when the SearchAgent parser uses provider tools.",
    )
    parser.add_argument(
        "--searchagent-rollout-model-name",
        type=str,
        default="default",
        help="Named slime/SGLang rollout model to call from SearchAgent.",
    )
    return parser

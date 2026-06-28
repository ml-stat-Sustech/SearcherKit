from __future__ import annotations


def add_searchagent_slime_fallback_arguments(parser):
    parser.add_argument(
        "--advantage-estimator",
        choices=("grpo", "igpo"),
        default="grpo",
        help=(
            "SearchAgent slime advantage estimator. 'igpo' uses the custom "
            "SearchAgent IGPO advantage path and does not enable slime PPO/critic."
        ),
    )
    return add_searchagent_slime_arguments(parser)


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
    parser.add_argument(
        "--searchagent-igpo-model-name",
        type=str,
        default=None,
        help=(
            "Named slime/SGLang model used to score IGPO answer likelihoods. "
            "Defaults to --searchagent-rollout-model-name."
        ),
    )
    parser.add_argument(
        "--searchagent-igpo-reward-coef",
        type=float,
        default=1.0,
        help="Coefficient applied to normalized IGPO token rewards before computing advantages.",
    )
    parser.add_argument(
        "--searchagent-igpo-outcome-reward-coef",
        type=float,
        default=1.0,
        help="Coefficient applied to the normalized scalar outcome reward in IGPO advantages.",
    )
    parser.add_argument(
        "--searchagent-truncation-penalty",
        type=float,
        default=-1.0,
        help="Reward penalty applied when a rollout hits SearchAgent context or length truncation.",
    )
    parser.add_argument(
        "--searchagent-eval-concurrency",
        type=int,
        default=64,
        help="Maximum concurrent eval samples for SearchAgent slime eval; <=0 disables the limit.",
    )
    return parser

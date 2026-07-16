from __future__ import annotations


def add_searcherkit_slime_fallback_arguments(parser):
    parser.add_argument(
        "--advantage-estimator",
        choices=("grpo", "igpo"),
        default="grpo",
        help=(
            "SearcherKit slime advantage estimator. 'igpo' uses the custom "
            "SearcherKit IGPO advantage path and does not enable slime PPO/critic."
        ),
    )
    return add_searcherkit_slime_arguments(parser)


def add_searcherkit_slime_arguments(parser):
    parser.add_argument(
        "--searcherkit-agent-config",
        type=str,
        default=None,
        help="YAML file containing the SearcherKit agent config used by slime rollouts.",
    )
    parser.add_argument(
        "--searcherkit-agent-config-key",
        type=str,
        default="agent",
        help="Dot path to the training AgentConfig inside --searcherkit-agent-config.",
    )
    parser.add_argument(
        "--searcherkit-eval-agent-config-key",
        type=str,
        default=None,
        help="Dot path to the eval AgentConfig; defaults to --searcherkit-agent-config-key.",
    )
    parser.add_argument(
        "--searcherkit-tool-call-parser",
        type=str,
        default="qwen",
        help="SGLang tool-call parser name used when the SearcherKit parser uses provider tools.",
    )
    parser.add_argument(
        "--searcherkit-reasoning-parser",
        type=str,
        default="qwen3",
        help="SGLang reasoning parser name used when the SearcherKit parser uses provider tools.",
    )
    parser.add_argument(
        "--searcherkit-rollout-model-name",
        type=str,
        default="default",
        help="Named slime/SGLang rollout model to call from SearcherKit.",
    )
    parser.add_argument(
        "--searcherkit-igpo-model-name",
        type=str,
        default=None,
        help=(
            "Named slime/SGLang model used to score IGPO answer likelihoods. "
            "Defaults to --searcherkit-rollout-model-name."
        ),
    )
    parser.add_argument(
        "--searcherkit-igpo-reward-coef",
        type=float,
        default=1.0,
        help="Coefficient applied to normalized IGPO token rewards before computing advantages.",
    )
    parser.add_argument(
        "--searcherkit-igpo-outcome-reward-coef",
        type=float,
        default=1.0,
        help="Coefficient applied to the normalized scalar outcome reward in IGPO advantages.",
    )
    parser.add_argument(
        "--searcherkit-igpo-reward-side",
        choices=("actor", "rollout"),
        default="rollout",
        help=(
            "Where SearcherKit IGPO answer-likelihood rewards are computed. "
            "'actor' computes them with the Megatron actor before advantage calculation; "
            "'rollout' keeps the older SGLang rollout-time scoring path."
        ),
    )
    parser.add_argument(
        "--searcherkit-igpo-actor-score-micro-batch-size",
        type=int,
        default=8,
        help="Micro-batch size for actor-side IGPO gold-answer logprob scoring.",
    )
    parser.add_argument(
        "--searcherkit-ppo-ratio-mode",
        choices=("token", "step"),
        default="token",
        help=(
            "PPO ratio granularity for SearcherKit slime policy loss. "
            "'step' aggregates contiguous valid-token spans before PPO clipping, "
            "matching AReal's importance_sampling_level=step more closely."
        ),
    )
    parser.add_argument(
        "--searcherkit-truncation-penalty",
        type=float,
        default=-1.0,
        help="Reward penalty applied when a rollout hits SearcherKit context or length truncation.",
    )
    parser.add_argument(
        "--searcherkit-answer-pattern",
        type=str,
        default=r"\\boxed\{(?P<answer>[^}]*)\}",
        help="Regex used to extract the final answer; must include a named 'answer' group.",
    )
    parser.add_argument(
        "--searcherkit-eval-concurrency",
        type=int,
        default=64,
        help="Maximum concurrent eval samples for SearcherKit slime eval; <=0 disables the limit.",
    )
    return parser

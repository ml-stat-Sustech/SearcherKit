from __future__ import annotations

import argparse
import os
import sys

from searchagent.training.slime_args import add_searchagent_slime_arguments


def _build_fallback_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m searchagent.training.train_slime",
        description=(
            "SearchAgent slime training entry. Full slime/Megatron/SGLang "
            "dependencies are required to parse and run the complete training CLI."
        ),
    )
    add_searchagent_slime_arguments(parser)
    parser.add_argument(
        "--show-missing-slime-deps",
        action="store_true",
        help="Show this fallback help when slime runtime dependencies are unavailable.",
    )
    return parser


def _parse_args(argv: list[str] | None):
    current_argv = sys.argv[1:] if argv is None else argv
    try:
        from slime.utils.arguments import parse_args
    except ModuleNotFoundError as exc:
        if any(arg in {"-h", "--help", "--show-missing-slime-deps"} for arg in current_argv):
            print(
                "Full slime CLI dependencies are not importable in this environment "
                f"({exc.name}). Showing SearchAgent-specific fallback help.\n",
                file=sys.stderr,
            )
            parser = _build_fallback_parser()
            parser.parse_args(current_argv)
            raise SystemExit(0) from exc
        raise RuntimeError(
            "slime training dependencies are not installed or not on PYTHONPATH. "
            "Install the THUDM/slime runtime requirements, including sglang_router, "
            "sglang, Ray, and Megatron, before running training."
        ) from exc

    if argv is None:
        return parse_args(add_searchagent_slime_arguments)

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        return parse_args(add_searchagent_slime_arguments)
    finally:
        sys.argv = old_argv


def train(args) -> None:
    import ray

    from slime.ray.placement_group import (
        create_placement_groups,
        create_rollout_manager,
        create_training_models,
    )
    from slime.utils.logging_utils import (
        finish_tracking,
        init_tracking,
        update_tracking_open_metrics,
    )
    from slime.utils.logging_utils import configure_logger
    from slime.utils.misc import should_run_periodic_action

    configure_logger()
    if not ray.is_initialized():
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "auto"),
            ignore_reinit_error=True,
        )
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def clear_train_memory(actor_trains_this_step: bool) -> None:
        if args.offload_train:
            return
        if not args.use_critic or actor_trains_this_step:
            actor_model.clear_memory()
        elif critic_model is not None:
            critic_model.clear_memory()

    def save(rollout_id: int) -> None:
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if actor_trains_this_step:
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic and critic_model is not None:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic and critic_model is not None:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        clear_train_memory(actor_trains_this_step)

        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())

        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


def main(argv: list[str] | None = None) -> None:
    train(_parse_args(argv))


if __name__ == "__main__":
    main()

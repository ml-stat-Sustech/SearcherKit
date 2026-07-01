from __future__ import annotations

import argparse
from typing import Sequence


PLUGIN_DESCRIPTIONS = {
    "local-wiki": "Read/preprocess MediaWiki dumps and deploy them to Elasticsearch.",
    "browsecomp-plus": "Read/preprocess BrowseComp Plus corpora and deploy them to Elasticsearch.",
}


def _clean_plugin_args(args: Sequence[str]) -> list[str]:
    cleaned = list(args)
    if cleaned and cleaned[0] == "--":
        cleaned = cleaned[1:]
    return cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="searchagent plugins",
        description="Discover and run SearchAgent plugin utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List bundled plugins")

    deploy = subparsers.add_parser(
        "deploy",
        help="Run a plugin deployment command, such as Elasticsearch indexing.",
    )
    deploy.add_argument("plugin", choices=sorted(PLUGIN_DESCRIPTIONS))
    deploy.add_argument(
        "plugin_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected plugin deploy command.",
    )
    convert = subparsers.add_parser(
        "convert",
        add_help=False,
        help="Run a plugin conversion command, such as SFT dataset conversion.",
    )
    convert.add_argument(
        "convert_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the conversion command.",
    )
    return parser


def _print_plugins() -> None:
    for name, description in PLUGIN_DESCRIPTIONS.items():
        print(f"{name}: {description}")


def _deploy(plugin: str, plugin_args: Sequence[str]) -> None:
    args = _clean_plugin_args(plugin_args)
    if plugin == "local-wiki":
        from searchagent.plugins.local_wiki.deploy_elasticsearch import main as deploy_main

        deploy_main(args, prog="searchagent plugins deploy local-wiki")
        return
    if plugin == "browsecomp-plus":
        from searchagent.plugins.browsecomp_plus.deploy_elasticsearch import main as deploy_main

        deploy_main(args, prog="searchagent plugins deploy browsecomp-plus")
        return
    raise ValueError(f"unknown plugin: {plugin}")


def _convert(convert_args: Sequence[str]) -> int:
    from searchagent.plugins.conversion.main import main as convert_main

    return convert_main(
        _clean_plugin_args(convert_args),
        prog="searchagent plugins convert",
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv is not None:
        argv = list(argv)
        if argv and argv[0] == "convert":
            return _convert(argv[1:])
    args = build_parser().parse_args(argv)
    if args.command == "list":
        _print_plugins()
        return 0
    if args.command == "deploy":
        _deploy(args.plugin, args.plugin_args)
        return 0
    if args.command == "convert":
        return _convert(args.convert_args)
    raise ValueError(f"unknown plugins command: {args.command}")

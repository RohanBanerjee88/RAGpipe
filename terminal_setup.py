"""Management commands which do not load inference models."""

import argparse
import os
from config import DEFAULT_COLLECTION, WEB_MAX_PAGES

from model_setup import (
    configure_session, model_status, prepare_models, profiles, save_selection,
)


def command_line():
    parser = argparse.ArgumentParser(description="Local collection knowledge assistant")
    parser.add_argument("--model", help="generation profile, Hub repository, or local directory")
    parser.add_argument("--offline", action="store_true", help="use local model files only")
    parser.add_argument("--collection", help="auto or one collection name")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("setup", help="guided one-time model setup")
    configuration = sub.add_parser("config", help="owner-only local settings").add_subparsers(dest="action", required=True)
    for action in ("init", "path", "check"):
        configuration.add_parser(action)
    models = sub.add_parser("models").add_subparsers(dest="action", required=True)
    models.add_parser("list")
    prepare = models.add_parser("prepare")
    prepare.add_argument("profile")
    collections = sub.add_parser("collections").add_subparsers(dest="action", required=True)
    collections.add_parser("list")
    for action in ("enable", "disable"):
        collections.add_parser(action).add_argument("name")
    importer = collections.add_parser("import")
    importer.add_argument("name")
    importer.add_argument("source", help="local file/directory, website, or GitHub tree/blob URL")
    importer.add_argument("--max-pages", type=int, default=WEB_MAX_PAGES,
                          help="maximum website pages or GitHub text files")
    importer.add_argument("--quiet", action="store_true", help="show a warning count instead of individual import warnings")
    args = parser.parse_args()
    if args.command == "config":
        from private_config import config_command
        config_command(args.action)
        return True
    configure_session(args.model, args.offline)
    os.environ["FAQ_COLLECTION"] = args.collection or os.getenv("FAQ_COLLECTION", DEFAULT_COLLECTION)
    if args.command == "models":
        if args.action == "prepare":
            prepare_models(args.profile)
        else:
            for profile in profiles()[1].values():
                print(f"{profile.name}: {model_status(profile)} | {profile.model or 'source excerpts'}")
        return True
    if args.command == "setup":
        default, available = profiles()
        for profile in available.values():
            print(f"{profile.name}: {model_status(profile)}")
        selection = input(f"Profile [{default}]: ").strip() or default
        if selection not in available:
            raise ValueError("Add a named profile to model_profiles.toml or the private config before guided setup")
        prepare_models(selection)
        save_selection(selection)
        print("Saved. Start with: python main.py")
        return True
    if args.command == "collections":
        from knowledge_store import import_collection, list_collections, set_enabled
        if args.action == "import":
            result = import_collection(args.name, args.source, max_pages=args.max_pages)
            print(f"{result['name']}: {len(result['records'])} records")
            warnings = result.get("warnings", [])
            if args.quiet and warnings:
                print(f"{len(warnings)} warnings retained in the collection manifest; rerun without --quiet for details.")
            elif warnings:
                for message in warnings[:20]:
                    print(f"Warning: {message}")
                if len(warnings) > 20:
                    print(f"{len(warnings) - 20} further warnings retained in the collection manifest.")
        elif args.action in {"enable", "disable"}:
            set_enabled(args.name, args.action == "enable")
            print(f"{args.name}: {args.action}d")
        else:
            for collection in list_collections(include_disabled=True):
                status = "enabled" if collection.get("enabled", True) else "disabled"
                print(f"{collection['name']}: {len(collection['records'])} records ({status})")
        return True
    return False

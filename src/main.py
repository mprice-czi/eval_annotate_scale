#!/usr/bin/env python3
"""
Simple Main Entry Point

Demo CLI application for the eval_annotate_scale project.
Verifies Python and Bazel integration is working correctly.
"""

import click


@click.command()
@click.option("--name", default="World", help="Name to greet")
def main(name: str) -> None:
    """Simple greeting program."""
    click.echo(f"Hello, {name}! Python 3.13.5 and Bazel are working!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple main entry point.
"""

import click


@click.command()
@click.option("--name", default="World", help="Name to greet")
def main(name):
    """Simple greeting program."""
    click.echo(f"Hello, {name}! Python 3.13.5 and Bazel are working!")


if __name__ == "__main__":
    main()
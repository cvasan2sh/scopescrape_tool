"""ScopeScrape CLI built on Click.

Commands:
    scan      - Run pain point analysis on specified platforms
    config    - Display current configuration
    platforms - List available platforms and API status
"""

from __future__ import annotations

from pathlib import Path

import click

from scopescrape import __version__
from scopescrape.config import load_config, validate_config
from scopescrape.log import setup_logging


@click.group()
@click.version_option(version=__version__, prog_name="scopescrape")
@click.option("--config", "config_path", type=click.Path(), default=None, help="Path to config YAML")
@click.option("--verbose", is_flag=True, help="Enable debug logging")
@click.option("--quiet", is_flag=True, help="Suppress info logging")
@click.pass_context
def main(ctx, config_path, verbose, quiet):
    """ScopeScrape: Community pain point discovery tool."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)
    ctx.obj["logger"] = setup_logging(verbose=verbose, quiet=quiet)


@main.command()
@click.option("--subreddits", type=str, default=None, help="Comma-separated subreddit names")
@click.option("--keywords", type=str, default=None, help="Comma-separated search terms")
@click.option("--platforms", type=click.Choice(["reddit", "hn", "github", "stackoverflow", "twitter", "producthunt", "indiehackers", "all"]), default=None)
@click.option("--time-range", type=click.Choice(["day", "week", "month", "year"]), default=None)
@click.option("--limit", type=int, default=None, help="Max posts to analyze")
@click.option("--min-score", type=float, default=None, help="Minimum composite score threshold")
@click.option("--output", "output_format", type=click.Choice(["json", "csv", "parquet", "airtable"]), default="json")
@click.option("--output-file", type=click.Path(), default=None, help="Destination file path")
@click.option("--dry-run", is_flag=True, help="Show what would be scanned without executing")
@click.pass_context
def scan(ctx, subreddits, keywords, platforms, time_range, limit, min_score, output_format, output_file, dry_run):
    """Run pain point analysis on community platforms."""
    config = ctx.obj["config"]
    logger = ctx.obj["logger"]

    # Apply CLI overrides to config
    scan_config = config.get("scan", {})
    platforms_list = _resolve_platforms(platforms, scan_config)
    effective_limit = limit or scan_config.get("default_limit", 100)
    effective_time_range = time_range or scan_config.get("default_time_range", "week")
    effective_min_score = min_score if min_score is not None else config.get("scoring", {}).get("min_score", 5.0)

    # Validate (public JSON adapter does not need Reddit API credentials)
    errors = validate_config(config, require_reddit=False)
    if errors:
        for err in errors:
            click.echo(f"Config error: {err}", err=True)
        raise click.Abort()

    if not subreddits and not keywords:
        click.echo("Error: provide --subreddits and/or --keywords", err=True)
        raise click.Abort()

    # Build query dict
    queries = {}
    if subreddits:
        queries["subreddits"] = [s.strip() for s in subreddits.split(",")]
    if keywords:
        queries["keywords"] = [k.strip() for k in keywords.split(",")]
    queries["limit"] = effective_limit
    queries["time_range"] = effective_time_range

    # Determine output file
    if output_file is None:
        output_file = f"results.{output_format}"

    logger.info(
        f"Scan: platforms={platforms_list}, queries={queries}, "
        f"min_score={effective_min_score}, output={output_file}"
    )

    if dry_run:
        click.echo("[DRY RUN] Would scan with the above parameters. Exiting.")
        return

    # Import pipeline here to avoid slow imports on --help
    from scopescrape.pipeline import Pipeline

    pipeline = Pipeline(config)
    pipeline.run(
        platforms=platforms_list,
        queries=queries,
        export_format=output_format,
        output_file=Path(output_file),
        min_score=effective_min_score,
    )

    click.echo(f"Done. Results written to {output_file}")


@main.command("config")
@click.pass_context
def show_config(ctx):
    """Display current configuration (credentials masked)."""
    import yaml

    config = ctx.obj["config"]

    # Mask sensitive values
    display = _mask_config(config)
    click.echo(yaml.dump(display, default_flow_style=False, sort_keys=False))


@main.command()
@click.pass_context
def platforms(ctx):
    """List available platforms and their API status."""
    config = ctx.obj["config"]

    platform_info = [
        ("reddit", "Reddit (via PRAW)", bool(config.get("reddit", {}).get("client_id"))),
        ("hn", "Hacker News (via Algolia)", True),  # No auth required
        ("github", "GitHub (via REST Search API)", True),  # No auth required (optional token)
        ("stackoverflow", "Stack Overflow (via Stack Exchange API)", True),  # No auth required (optional key)
        ("twitter", "Twitter/X (via Nitter HTML scraping)", True),  # No auth required
        ("producthunt", "Product Hunt (via GraphQL API)", True),  # No auth required (optional token)
        ("indiehackers", "Indie Hackers (via Algolia)", True),  # No auth required
    ]

    click.echo("Available platforms:\n")
    for key, name, ready in platform_info:
        status = click.style("ready", fg="green") if ready else click.style("not configured", fg="red")
        click.echo(f"  {key:10s} {name:35s} [{status}]")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8888, type=int, help="Port to serve on")
def web(host, port):
    """Launch the ScopeScrape web UI (local server)."""
    try:
        from scopescrape.web import start_server
    except ImportError:
        click.echo(
            "Web dependencies not installed. Run:\n"
            '  pip install -e ".[webapp]"',
            err=True,
        )
        raise click.Abort()

    start_server(host=host, port=port)


def _resolve_platforms(cli_value: str | None, scan_config: dict) -> list[str]:
    """Resolve platform selection from CLI flag or config default."""
    if cli_value == "all":
        return ["reddit", "hn"]
    if cli_value:
        return [cli_value]
    return scan_config.get("default_platforms", ["reddit"])


def _mask_config(config: dict) -> dict:
    """Mask sensitive values for display."""
    masked = {}
    sensitive_keys = {"client_id", "client_secret", "password", "token", "secret"}

    for key, value in config.items():
        if isinstance(value, dict):
            masked[key] = _mask_config(value)
        elif key in sensitive_keys and isinstance(value, str) and len(value) > 4:
            masked[key] = value[:4] + "****"
        else:
            masked[key] = value

    return masked

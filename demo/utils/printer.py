from rich.console import Console
from rich.panel import Panel

console = Console()

def section(title: str):
    console.rule(f"[bold cyan]{title}")

def success(msg: str):
    console.print(f"[bold green]✔ {msg}")

def warning(msg: str):
    console.print(f"[bold yellow]{msg}")

def error(msg: str):
    console.print(f"[bold red]{msg}")

def info(msg: str):
    console.print(msg)

def panel(title, text):
    console.print(
        Panel(
            text,
            title=title,
            expand=False
        )
    )
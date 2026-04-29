import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text
from vaml.env.make import make_env


def play(env_name: str, seed: int = 42):
    console = Console()
    env = make_env(env_name, 1, seed, None)
    console.print(env.max_turns)

    console.print(Rule(f"[bold cyan]{env_name.upper()}[/bold cyan]"))
    console.print(
        Panel(
            env.instructions(),
            title="[bold]Instructions[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print("[dim]Type 'quit' to exit.[/dim]\n")

    idx = np.array([0], dtype=np.int32)
    obs, metrics = env.reset(idx)

    total_reward = 0.0

    while True:
        try:
            action = Prompt.ask(
                "[bold magenta]>[/bold magenta]", console=console
            )
        except EOFError, KeyboardInterrupt:
            console.print()
            break

        if action.strip().lower() == "quit":
            break

        obs, rewards, dones, metrics = env.step(idx, [action])
        reward = rewards[0]
        done = dones[0]
        total_reward += reward

        console.print(metrics)

        if reward > 0:
            console.print(
                f"[bold yellow]reward[/bold yellow] [yellow]+{reward:.2f}[/yellow]"
            )

        if done:
            color = "green" if total_reward > 0 else "red"
            console.print(
                Panel(
                    Text.assemble(
                        ("total reward ", "bold"),
                        (f"{total_reward:.2f}", f"bold {color}"),
                    ),
                    border_style=color,
                    padding=(0, 1),
                )
            )
            total_reward = 0.0

        console.print(obs[0])

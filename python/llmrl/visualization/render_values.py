#!/usr/bin/env python3
"""Render values.np file to a video showing value function evolution over training."""

import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def render_values_video(
    values_path: str,
    output_path: str = "values.mp4",
    fps: int = 10,
    dpi: int = 100,
    figsize: tuple[int, int] = (12, 6),
):
    """Render a values.np file to a video.

    Args:
        values_path: Path to the values.np or values.np.npy file
        output_path: Output video path (e.g., values.mp4)
        fps: Frames per second for the video
        dpi: DPI for the video resolution
        figsize: Figure size (width, height) in inches
    """
    # Load the values
    values = np.load(values_path)
    num_steps, seq_length = values.shape

    print(f"Loaded values with shape: {values.shape}")
    print(f"  Training steps: {num_steps}")
    print(f"  Sequence length: {seq_length}")

    # Find the global min/max for consistent y-axis
    vmin, vmax = -0.1, 1 #values.min(), values.max()
    margin = (vmax - vmin) * 0.1
    ymin, ymax = vmin - margin, vmax + margin

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Initialize the line
    positions = np.arange(seq_length)
    (line,) = ax.plot(positions, values[0], "b-", linewidth=1.5)

    ax.set_xlim(0, seq_length - 1)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Position in Sequence")
    ax.set_ylabel("Value")
    title = ax.set_title(f"Value Function - Step 0/{num_steps}")
    ax.grid(True, alpha=0.3)

    def update(frame):
        line.set_ydata(values[frame])
        title.set_text(f"Value Function - Step {frame + 1}/{num_steps}")
        return line, title

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=num_steps, interval=1000 // fps, blit=True
    )

    # Save to video
    print(f"Rendering video to {output_path}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_path, writer=writer, dpi=dpi)
    print(f"Video saved to {output_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Render values.np file to a video showing value function evolution"
    )
    parser.add_argument(
        "values_path", type=str, help="Path to the values.np or values.np.npy file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="values.mp4",
        help="Output video path (default: values.mp4)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second (default: 10)"
    )
    parser.add_argument(
        "--dpi", type=int, default=100, help="Video DPI/resolution (default: 100)"
    )
    parser.add_argument(
        "--width", type=int, default=12, help="Figure width in inches (default: 12)"
    )
    parser.add_argument(
        "--height", type=int, default=6, help="Figure height in inches (default: 6)"
    )

    args = parser.parse_args()

    render_values_video(
        values_path=args.values_path,
        output_path=args.output,
        fps=args.fps,
        dpi=args.dpi,
        figsize=(args.width, args.height),
    )


if __name__ == "__main__":
    main()

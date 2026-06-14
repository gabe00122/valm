<script lang="ts">
	import type { HTMLInputAttributes } from "svelte/elements";
	import { cn } from "$lib/utils.js";

	type SliderProps = Omit<HTMLInputAttributes, "children" | "type" | "value"> & {
		ref?: HTMLInputElement | null;
		value?: number;
		orientation?: "horizontal" | "vertical";
		type?: "single" | "multiple" | "range";
	};

	let {
		ref = $bindable(null),
		value = $bindable(0),
		orientation = "horizontal",
		class: className,
		type: _type,
		min = 0,
		max = 100,
		style,
		...restProps
	}: SliderProps = $props();

	const toNumber = (input: string | number | null | undefined, fallback: number) => {
		const number = Number(input);
		return Number.isFinite(number) ? number : fallback;
	};

	let minValue = $derived(toNumber(min, 0));
	let maxValue = $derived(toNumber(max, 100));
	let currentValue = $derived(toNumber(value, minValue));
	let progress = $derived(
		maxValue === minValue
			? 0
			: Math.min(100, Math.max(0, ((currentValue - minValue) / (maxValue - minValue)) * 100))
	);
	let sliderStyle = $derived(`--slider-progress: ${progress}%; ${style ?? ""}`);
</script>

<input
	bind:this={ref}
	bind:value
	data-slot="slider"
	data-orientation={orientation}
	type="range"
	{min}
	{max}
	style={sliderStyle}
	class={cn(
		"relative block w-full touch-none appearance-none bg-transparent select-none disabled:cursor-not-allowed disabled:opacity-50 data-[orientation=vertical]:h-full data-[orientation=vertical]:min-h-40 data-[orientation=vertical]:w-4",
		className
	)}
	{...restProps}
/>

<style>
	input[type="range"][data-slot="slider"] {
		--slider-track-color: var(--muted);
		--slider-range-color: var(--primary);
		--slider-thumb-ring: color-mix(in oklab, var(--ring) 50%, transparent);
		cursor: pointer;
	}

	input[type="range"][data-slot="slider"][data-orientation="horizontal"] {
		height: 1.25rem;
		background: linear-gradient(
			to right,
			var(--slider-range-color) var(--slider-progress),
			var(--slider-track-color) var(--slider-progress)
		);
		background-clip: content-box;
		padding-block: 0.5rem;
	}

	input[type="range"][data-slot="slider"][data-orientation="vertical"] {
		width: 1.25rem;
		background: linear-gradient(
			to top,
			var(--slider-range-color) var(--slider-progress),
			var(--slider-track-color) var(--slider-progress)
		);
		background-clip: content-box;
		direction: rtl;
		padding-inline: 0.5rem;
		writing-mode: vertical-lr;
	}

	input[type="range"][data-slot="slider"]::-webkit-slider-runnable-track {
		height: 0.25rem;
		background: transparent;
	}

	input[type="range"][data-slot="slider"][data-orientation="vertical"]::-webkit-slider-runnable-track {
		width: 0.25rem;
		height: 100%;
	}

	input[type="range"][data-slot="slider"]::-webkit-slider-thumb {
		width: 0.75rem;
		height: 0.75rem;
		margin-top: -0.25rem;
		appearance: none;
		background: white;
		border: 1px solid var(--ring);
		border-radius: 0;
		transition:
			color 150ms,
			box-shadow 150ms;
	}

	input[type="range"][data-slot="slider"][data-orientation="vertical"]::-webkit-slider-thumb {
		margin-top: 0;
		margin-inline-start: -0.25rem;
	}

	input[type="range"][data-slot="slider"]:hover::-webkit-slider-thumb,
	input[type="range"][data-slot="slider"]:active::-webkit-slider-thumb,
	input[type="range"][data-slot="slider"]:focus-visible::-webkit-slider-thumb {
		box-shadow: 0 0 0 1px var(--slider-thumb-ring);
	}

	input[type="range"][data-slot="slider"]:focus-visible {
		outline: none;
	}

	input[type="range"][data-slot="slider"]::-moz-range-track {
		height: 0.25rem;
		background: var(--slider-track-color);
		border: 0;
		border-radius: 0;
	}

	input[type="range"][data-slot="slider"]::-moz-range-progress {
		height: 0.25rem;
		background: var(--slider-range-color);
		border-radius: 0;
	}

	input[type="range"][data-slot="slider"]::-moz-range-thumb {
		width: 0.75rem;
		height: 0.75rem;
		background: white;
		border: 1px solid var(--ring);
		border-radius: 0;
		transition:
			color 150ms,
			box-shadow 150ms;
	}

	input[type="range"][data-slot="slider"]:hover::-moz-range-thumb,
	input[type="range"][data-slot="slider"]:active::-moz-range-thumb,
	input[type="range"][data-slot="slider"]:focus-visible::-moz-range-thumb {
		box-shadow: 0 0 0 1px var(--slider-thumb-ring);
	}
</style>

<script lang="ts">
    import { tick } from "svelte";
    import type { Episode } from "$lib/episodes";

    interface Props {
        episode: Episode;
        selectedIndex: number | null;
        hoveredIndex: number | null;
        metricKey: string;
    }

    type MetricValues = ArrayLike<number>;

    let {
        episode,
        selectedIndex = $bindable(null),
        hoveredIndex = $bindable(null),
        metricKey: metricKey,
    }: Props = $props();

    let tokenElements = $state<HTMLElement[]>([]);
    let metricValues = $derived(getMetricValues(episode, metricKey));
    let metricRange = $derived(getMetricRange(metricValues));

    function getMetricValues(
        episode: Episode,
        metricKey: string,
    ): MetricValues | null {
        if (metricKey === "none") {
            return null;
        }

        return episode.tokenMetrics[metricKey] ?? null;
    }

    function getMetricRange(values: MetricValues | null) {
        if (values === null) {
            return null;
        }

        let min = Infinity;
        let max = -Infinity;

        for (let i = 0; i < values.length; i++) {
            const value = values[i];

            if (!Number.isFinite(value)) {
                continue;
            }

            min = Math.min(min, value);
            max = Math.max(max, value);
        }

        return min === Infinity || min === max ? null : { min, max };
    }

    function getHue(value: number | undefined) {
        if (value === undefined || !Number.isFinite(value)) {
            return "transparent";
        }

        if (metricRange === null) {
            return "color-mix(in srgb, var(--token-color) 18%, transparent)";
        }

        const ratio = Math.min(
            Math.max(
                (value - metricRange.min) / (metricRange.max - metricRange.min),
                0,
            ),
            1,
        );
        const hue = (1 - ratio) * 240;
        return `hsl(${hue}, 50%, 50%)`;
    }

    function selectToken(index: number) {
        selectedIndex = selectedIndex === index ? null : index;
    }

    function hoverToken(index: number | null) {
        hoveredIndex = index;
    }

    function handleKeydown(event: KeyboardEvent, index: number) {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            selectToken(index);
        }
    }

    $effect(() => {
        const index = selectedIndex;
        const tokenCount = episode.tokens.length;

        if (index === null || index < 0 || index >= tokenCount) {
            return;
        }

        tick().then(() => {
            tokenElements[index]?.scrollIntoView({
                block: "nearest",
                inline: "nearest",
            });
        });
    });
</script>

<div class="tokens font-mono whitespace-pre-wrap leading-tight">
    {#each episode.tokens as token, index}
        <span
            bind:this={tokenElements[index]}
            tabindex="0"
            role="button"
            aria-pressed={selectedIndex === index}
            class:hovered={hoveredIndex === index && selectedIndex !== index}
            class:selected={selectedIndex === index}
            style="--viz-token-color: {getHue(metricValues?.[index])};"
            onclick={() => selectToken(index)}
            onpointerenter={() => hoverToken(index)}
            onpointerleave={() => hoverToken(null)}
            onkeydown={(event) => handleKeydown(event, index)}>{token}</span
        >
    {/each}
</div>

<style>
    .tokens span {
        background-color: var(--viz-token-color);
        cursor: pointer;
    }

    .tokens span:hover {
        background-color: var(--foreground);
        color: var(--background);
    }

    .tokens span.hovered {
        background-color: var(--foreground);
        color: var(--background);
    }

    .tokens span.selected {
        background: color-mix(in srgb, var(--token-color) 45%, transparent);
    }

    .tokens span:focus-visible {
        outline: 2px solid var(--token-color);
        outline-offset: 2px;
    }
</style>

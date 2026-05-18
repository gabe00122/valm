<script lang="ts">
    import type { Episode } from "$lib/episodes";

    interface Props {
        episode: Episode;
        selectedIndex: number | null;
        metricKey: string;
    }

    type MetricValues = ArrayLike<number>;

    let {
        episode,
        selectedIndex = $bindable(null),
        metricKey: metricKey,
    }: Props = $props();

    // let selectedIndex = $state<number | null>(null);

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

    function handleKeydown(event: KeyboardEvent, index: number) {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            selectToken(index);
        }
    }
</script>

<div class="tokens font-mono whitespace-pre-wrap leading-tight">
    {#each episode.tokens as token, index}
        <span
            tabindex="0"
            role="button"
            aria-pressed={selectedIndex === index}
            class:selected={selectedIndex === index}
            style="--viz-token-color: {getHue(metricValues?.[index])};"
            onclick={() => selectToken(index)}
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
        background-color: color-mix(
            in srgb,
            var(--token-color) 25%,
            transparent
        );
    }

    .tokens span.selected {
        background: color-mix(in srgb, var(--token-color) 45%, transparent);
    }

    .tokens span:focus-visible {
        outline: 2px solid var(--token-color);
        outline-offset: 2px;
    }
</style>

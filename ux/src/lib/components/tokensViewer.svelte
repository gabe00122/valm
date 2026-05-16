<script lang="ts">
    import type { Episode } from "$lib/episodes";

    interface Props {
        episode: Episode;
        selectedIndex: number | null;
    }

    let { episode, selectedIndex = $bindable(null) }: Props = $props();

    // let selectedIndex = $state<number | null>(null);

    function getHue(value: number) {
        const ratio = value;
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
            style="--viz-token-color: {getHue(episode.logProbs[index])};"
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

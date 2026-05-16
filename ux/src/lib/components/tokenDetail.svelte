<script lang="ts">
    import type { Episode } from "$lib/episodes";

    interface Props {
        selectedIndex: number | null;
        episode: Episode;
    }

    let { selectedIndex, episode }: Props = $props();

    let updateEntries = $derived(Object.entries(episode.updateMetrics));
</script>

<div>
    {#if selectedIndex !== null}
        <div>Token Index: {selectedIndex}</div>
        <div>Log Prob: {episode.logProbs[selectedIndex]}</div>
        {#each updateEntries as [name, value]}
            <div>{name}: {value[selectedIndex]}</div>
        {/each}
    {/if}
</div>

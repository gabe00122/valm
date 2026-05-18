<script lang="ts">
    import type { Episode } from "$lib/episodes";

    interface Props {
        selectedIndex: number | null;
        episode: Episode;
    }

    type DetailRow = [label: string, value: string | number];

    let { selectedIndex, episode }: Props = $props();

    function formatNumber(value: number | undefined) {
        return value === undefined || Number.isNaN(value)
            ? "n/a"
            : value.toPrecision(6);
    }

    function detailRows(index: number): DetailRow[] {
        return [
            ["token", episode.tokens[index] ?? "n/a"],
            ...Object.entries(episode.tokenMetrics).map<DetailRow>(
                ([name, value]) => [name, formatNumber(value[index])],
            ),
        ];
    }
</script>

{#snippet detailRow(label: string, value: string | number)}
    <div
        class="grid grid-cols-[minmax(5.5rem,0.8fr)_minmax(0,1fr)] items-baseline gap-2"
    >
        <span class="text-current/70">{label}</span>
        <code class="min-w-0 wrap-anywhere font-mono">{value}</code>
    </div>
{/snippet}

<div class="grid gap-2 p-3 text-sm">
    {#if selectedIndex !== null}
        {#each detailRows(selectedIndex) as [label, value]}
            {@render detailRow(label, value)}
        {/each}
    {:else}
        <div class="text-current/60">Select a token</div>
    {/if}
</div>

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

    function formatPolicyMask(value: number | undefined) {
        if (value === undefined) {
            return "n/a";
        }

        return value === 0 ? "false" : "true";
    }

    function detailRows(index: number): DetailRow[] {
        return [
            ["Token", episode.tokens[index] ?? "n/a"],
            ["Token ID", episode.tokenIds[index] ?? "n/a"],
            ["Token Index", index],
            ["Log Prob", formatNumber(episode.logProbs[index])],
            ["Value", formatNumber(episode.values[index])],
            ["Reward", formatNumber(episode.rewards[index])],
            ["Policy Mask", formatPolicyMask(episode.policyMask[index])],
            ...Object.entries(episode.updateMetrics).map<DetailRow>(
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

<script lang="ts">
    import type { Episode } from "$lib/episodes";

    interface Probs {
        episode: Episode;
        metricKey: string;
        selectedIndex: number | null;
    }

    let { episode, metricKey, selectedIndex } = $props();
</script>

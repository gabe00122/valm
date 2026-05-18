<script lang="ts">
    import { getEpisode } from "./episode.remote";
    import TokenViewer from "$lib/components/tokensViewer.svelte";
    import * as Resizable from "$lib/components/ui/resizable";
    import { ScrollArea } from "$lib/components/ui/scroll-area/index.js";
    import { decodeEpisodes } from "$lib/decode";
    import TokenDetail from "$lib/components/tokenDetail.svelte";
    import Separator from "$lib/components/ui/separator/separator.svelte";
    import ShowControl from "$lib/components/showControl.svelte";
    import MetricGraph from "$lib/components/metricGraph.svelte";

    let episodeId = $state(0);
    let selectedIndex: number | null = $state(0);
    let hoveredIndex: number | null = $state(null);
    let metricKey: string = $state("none");
    let episode = $derived(decodeEpisodes(await getEpisode(episodeId)));

    $effect(() => {
        episodeId;
        selectedIndex = 0;
        hoveredIndex = null;
    });
</script>

<Resizable.PaneGroup direction="horizontal">
    <Resizable.Pane defaultSize={0.2}>
        Episode:
        <input type="number" bind:value={episodeId} />
        <Separator />
        <ShowControl bind:metricKey {episode} />
        <Separator />
        <TokenDetail {selectedIndex} {episode} />
    </Resizable.Pane>
    <Resizable.Handle />
    <Resizable.Pane defaultSize={0.8}>
        <Resizable.PaneGroup direction="vertical">
            <Resizable.Pane defaultSize={25}>
                <MetricGraph
                    {episode}
                    bind:selectedIndex
                    bind:hoveredIndex
                    {metricKey}
                />
            </Resizable.Pane>
            <Resizable.Handle />
            <Resizable.Pane defaultSize={75}>
                <div class="h-full min-h-0 overflow-hidden">
                    <ScrollArea class="h-full">
                        <TokenViewer
                            bind:selectedIndex
                            bind:hoveredIndex
                            {episode}
                            {metricKey}
                        />
                    </ScrollArea>
                </div>
            </Resizable.Pane>
        </Resizable.PaneGroup>
    </Resizable.Pane>
</Resizable.PaneGroup>

<script lang="ts">
    import { getEpisode, getRunInfo, getRuns } from "./episode.remote";
    import TokenViewer from "$lib/components/tokensViewer.svelte";
    import * as Resizable from "$lib/components/ui/resizable";
    import { ScrollArea } from "$lib/components/ui/scroll-area/index.js";
    import * as Select from "$lib/components/ui/select/index.js";
    import { Slider } from "$lib/components/ui/slider";
    import { decodeEpisodes } from "$lib/decode";
    import TokenDetail from "$lib/components/tokenDetail.svelte";
    import Separator from "$lib/components/ui/separator/separator.svelte";
    import ShowControl from "$lib/components/showControl.svelte";
    import MetricGraph from "$lib/components/metricGraph.svelte";

    let runs = $derived(await getRuns());
    let selectedRun: string = $state("");
    let run = $derived(selectedRun || runs[0]);
    let episodeCount = $derived((await getRunInfo(run)).episodeCount);

    let episodeId = $state(0);
    let selectedIndex: number | null = $state(0);
    let hoveredIndex: number | null = $state(null);
    let metricKey: string = $state("none");
    let episode = $derived(
        decodeEpisodes(
            await getEpisode({
                run,
                episodeId: Math.min(episodeId, episodeCount - 1),
            }),
        ),
    );

    $effect(() => {
        run;
        episodeId = 0;
    });

    $effect(() => {
        episodeId;
        selectedIndex = 0;
        hoveredIndex = null;
    });
</script>

<Resizable.PaneGroup direction="horizontal">
    <Resizable.Pane defaultSize={0.2}>
        <div class="flex h-full min-h-0 flex-col gap-3 p-3 text-sm">
            <div class="grid gap-1.5">
                Run:
                <Select.Root
                    type="single"
                    name="trainingRun"
                    bind:value={selectedRun}
                >
                    <Select.Trigger class="w-full">{run}</Select.Trigger>
                    <Select.Content>
                        {#each runs as runName (runName)}
                            <Select.Item value={runName} label={runName}>
                                {runName}
                            </Select.Item>
                        {/each}
                    </Select.Content>
                </Select.Root>
            </div>
            <Separator />
            <div class="grid gap-1.5">
                Episode: {episodeId} / {episodeCount - 1}
                <Slider
                    type="single"
                    bind:value={episodeId}
                    min={0}
                    max={Math.max(episodeCount - 1, 0)}
                    step={1}
                />
            </div>
            <Separator />
            <ShowControl bind:metricKey {episode} />
            <Separator />
            <TokenDetail {selectedIndex} {episode} />
        </div>
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

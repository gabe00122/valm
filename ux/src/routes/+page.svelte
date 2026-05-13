<script lang="ts">
    import { getEpisode } from "./episode.remote";
    import TokenViewer from "$lib/components/tokensViewer.svelte";
    import GraphPanel from "./components/graphPanel.svelte";
    import Button from "$lib/components/ui/button/button.svelte";
    import * as Resizable from "$lib/components/ui/resizable";
    import { ScrollArea } from "$lib/components/ui/scroll-area/index.js";

    let episodeId = $state(0);
</script>

<!-- <div class="root">
    <div class="sidePanel"></div>
    <div class="mainPanel">
        <div class="graphPanel"></div>
        <div class="tokenPanel"></div>
    </div>
</div> -->

<!-- <Button>Hello</Button>

<p>Episode Viewer</p>

<input type="number" bind:value={episodeId} />
 -->

<Resizable.PaneGroup direction="horizontal">
    <Resizable.Pane defaultSize={0.2}>
        <div class="flex h-[200px] items-center justify-center p-6">
            <span class="font-semibold">One</span>
        </div>
    </Resizable.Pane>
    <Resizable.Handle />
    <Resizable.Pane defaultSize={0.8}>
        <Resizable.PaneGroup direction="vertical">
            <Resizable.Pane defaultSize={25}>
                <div class="flex h-full items-center justify-center p-6">
                    <span class="font-semibold">Two</span>
                </div>
            </Resizable.Pane>
            <Resizable.Handle />
            <Resizable.Pane defaultSize={75}>
                <div class="h-full min-h-0 overflow-hidden">
                    <ScrollArea class="h-full">
                        <TokenViewer
                            tokens={(await getEpisode(episodeId)).tokens}
                        />
                    </ScrollArea>
                </div>
            </Resizable.Pane>
        </Resizable.PaneGroup>
    </Resizable.Pane>
</Resizable.PaneGroup>

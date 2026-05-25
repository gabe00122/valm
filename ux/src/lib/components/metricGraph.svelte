<script lang="ts">
    import * as Chart from "$lib/components/ui/chart/index.js";
    import type { Episode } from "$lib/episodes";
    import { Highlight, LineChart, type ChartState } from "layerchart";

    interface Props {
        episode: Episode;
        metricKey: string;
        selectedIndex: number | null;
        hoveredIndex: number | null;
    }

    type MetricDatum = {
        index: number;
        value: number;
    };

    let {
        episode,
        metricKey,
        selectedIndex = $bindable(null),
        hoveredIndex = $bindable(null),
    }: Props = $props();

    let context = $state<ChartState>(null!);
    let metricValues = $derived(getMetricValues(episode, metricKey));
    let chartData = $derived(getChartData(metricValues));
    let hoveredDatum = $derived(getDatum(hoveredIndex));
    let selectedDatum = $derived(getDatum(selectedIndex));
    let chartConfig = $derived({
        value: {
            label: metricKey === "none" ? "Metric" : metricKey,
            color: "var(--chart-2)",
        },
    } satisfies Chart.ChartConfig);
    let activeSeries = $derived([
        {
            key: "value",
            label: chartConfig.value.label,
            color: chartConfig.value.color,
        },
    ]);

    function getMetricValues(episode: Episode, metricKey: string) {
        if (metricKey === "none") {
            return null;
        }

        return episode.tokenMetrics[metricKey] ?? null;
    }

    function getChartData(values: ArrayLike<number> | null): MetricDatum[] {
        if (values === null) {
            return [];
        }

        const out: MetricDatum[] = [];
        for (let i = 0; i < values.length; i++) {
            out.push({
                index: i,
                value: values[i],
            });
        }

        return out;
    }

    function getDatum(index: number | null) {
        if (index === null) {
            return undefined;
        }

        return chartData[index];
    }

    function tooltipIndex() {
        return (
            (context?.tooltipState.data as Partial<MetricDatum> | null)
                ?.index ?? null
        );
    }

    function updateHoverIndex() {
        hoveredIndex = tooltipIndex();
    }

    function selectDatum() {
        const index = tooltipIndex();

        if (typeof index !== "number") {
            return;
        }

        selectedIndex = selectedIndex === index ? null : index;
    }
</script>

<div class="metric-graph h-full">
    {#if metricValues === null}
        <div
            class="text-muted-foreground flex h-full items-center justify-center text-xs"
        >
            No metric selected
        </div>
    {:else if chartData.length === 0}
        <div
            class="text-muted-foreground flex h-full items-center justify-center text-xs"
        >
            No metric data
        </div>
    {:else}
        <Chart.Container
            config={chartConfig}
            class="aspect-auto h-full"
            onclick={selectDatum}
            onmousemove={updateHoverIndex}
        >
            <LineChart
                bind:context
                data={chartData}
                x="index"
                y="value"
                series={activeSeries}
                padding={{ left: 40, top: 20, bottom: 20, right: 10 }}
            >
                {#snippet highlight()}
                    <Highlight
                        data={selectedDatum}
                        axis="x"
                        points={false}
                        motion="none"
                        lines={{
                            class: "metric-selected-focus-line",
                        }}
                    />
                    <Highlight
                        data={hoveredDatum}
                        axis="x"
                        points={false}
                        motion="none"
                        lines={{
                            class: "metric-hovered-focus-line",
                        }}
                    />
                {/snippet}
            </LineChart>
        </Chart.Container>
    {/if}
</div>

<style>
    :global(.metric-graph .metric-selected-focus-line) {
        stroke: var(--foreground);
        stroke-width: 1.75px;
    }

    :global(.metric-graph .metric-hovered-focus-line) {
        stroke: var(--foreground);
        stroke-width: 1.5px;
        stroke-dasharray: 4 4;
    }
</style>

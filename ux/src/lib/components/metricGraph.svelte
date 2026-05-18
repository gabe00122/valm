<script lang="ts">
    import * as Chart from "$lib/components/ui/chart/index.js";
    import type { Episode } from "$lib/episodes";
    import { Highlight, LineChart } from "layerchart";

    interface Props {
        episode: Episode;
        metricKey: string;
        selectedIndex: number | null;
        hoveredIndex: number | null;
    }

    type MetricDatum = {
        index: number;
        value: number | null;
    };

    let {
        episode,
        metricKey,
        selectedIndex = $bindable(null),
        hoveredIndex = $bindable(null),
    }: Props = $props();

    let chartContext = $state<any>();
    let metricValues = $derived(getMetricValues(episode, metricKey));
    let chartData = $derived(getChartData(episode, metricValues));
    let hoveredDatum = $derived(getDatum(hoveredIndex));
    let selectedDatum = $derived(getDatum(selectedIndex));
    let activeDatum = $derived(getActiveDatum());
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

    function getChartData(
        episode: Episode,
        values: ArrayLike<number> | null,
    ): MetricDatum[] {
        if (values === null) {
            return [];
        }

        return episode.tokens.map((token, index) => {
            const value = values[index];

            return {
                index,
                value: Number.isFinite(value) ? value : null,
            };
        });
    }

    function getDatum(index: number | null) {
        if (index === null) {
            return undefined;
        }

        return chartData[index];
    }

    function getActiveDatum() {
        return hoveredDatum ?? selectedDatum;
    }

    function formatIndex(index: number) {
        return index.toLocaleString();
    }

    function formatValue(value: unknown) {
        return typeof value === "number" && Number.isFinite(value)
            ? value.toPrecision(6)
            : "n/a";
    }

    function selectDatum(data: unknown) {
        const index =
            (data as Partial<MetricDatum> | null)?.index ?? hoveredIndex;

        if (typeof index !== "number") {
            return;
        }

        selectedIndex = selectedIndex === index ? null : index;
    }

    function updateHoveredIndex(data: unknown) {
        const index = (data as Partial<MetricDatum> | null)?.index;
        hoveredIndex = typeof index === "number" ? index : null;
    }

    $effect(() => {
        updateHoveredIndex(chartContext?.tooltip.data);
    });
</script>

<div class="metric-graph grid h-full min-h-0 grid-rows-[auto_1fr] border-b">
    <div class="flex items-center justify-between gap-3 px-3 py-2 text-xs">
        <div class="min-w-0">
            <div class="truncate font-medium">
                {metricKey === "none" ? "Select a metric" : metricKey}
            </div>
            <div class="text-muted-foreground">
                {#if activeDatum}
                    token {formatIndex(activeDatum.index)}: {formatValue(
                        activeDatum.value,
                    )}
                {:else}
                    {chartData.length.toLocaleString()} tokens
                {/if}
            </div>
        </div>
    </div>

    <div class="min-h-0 px-2 pb-2">
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
                class="aspect-auto h-full w-full"
                onclick={() => selectDatum(activeDatum)}
            >
                <LineChart
                    bind:context={chartContext}
                    data={chartData}
                    x="index"
                    y="value"
                    axis="x"
                    series={activeSeries}
                    props={{
                        spline: {
                            strokeWidth: 2,
                            motion: "none",
                            defined: (d: MetricDatum) => d.value !== null,
                        },
                        xAxis: {
                            format: (v: number) => formatIndex(v),
                        },
                        tooltip: {
                            context: {
                                onpointerleave: () => {
                                    hoveredIndex = null;
                                },
                            },
                        },
                    }}
                >
                    {#snippet tooltip()}
                        <Chart.Tooltip
                            hideLabel={false}
                            labelFormatter={(value) =>
                                typeof value === "number"
                                    ? formatIndex(value)
                                    : `${value}`}
                        />
                    {/snippet}
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
</div>

<style>
    :global(.metric-graph .metric-selected-focus-line) {
        stroke: var(--foreground) !important;
        stroke-width: 1.75px !important;
    }

    :global(.metric-graph .metric-hovered-focus-line) {
        stroke: var(--foreground) !important;
        stroke-width: 1.5px !important;
        stroke-dasharray: 4 4;
    }
</style>

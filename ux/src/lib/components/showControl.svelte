<script lang="ts">
    import * as Select from "$lib/components/ui/select/index.js";
    import type { Episode } from "$lib/episodes";

    interface Props {
        viewMetricKey: string;
        episode: Episode;
    }

    type MetricOption = {
        value: string;
        label: string;
    };

    let { viewMetricKey = $bindable("none"), episode }: Props = $props();

    let metricOptions = $derived<MetricOption[]>([
        { value: "none", label: "None" },
        { value: "logProbs", label: "Log Prob" },
        { value: "values", label: "Value" },
        { value: "rewards", label: "Reward" },
        { value: "policyMask", label: "Policy Mask" },
        ...Object.keys(episode.updateMetrics).map((name) => ({
            value: `updateMetrics:${name}`,
            label: name,
        })),
    ]);

    let selectedLabel = $derived(
        metricOptions.find((option) => option.value === viewMetricKey)?.label ??
            "None",
    );
</script>

<Select.Root
    type="single"
    name="tokenHighlightMetric"
    bind:value={viewMetricKey}
>
    <Select.Trigger class="w-full">{selectedLabel}</Select.Trigger>
    <Select.Content>
        {#each metricOptions as option (option.value)}
            <Select.Item value={option.value} label={option.label}>
                {option.label}
            </Select.Item>
        {/each}
    </Select.Content>
</Select.Root>

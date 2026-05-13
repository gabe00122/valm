<script lang="ts">
    interface Props {
        tokens: string[];
    }

    function getHue(index: number) {
        const pallet = ["#0b6a88", "#ec015a", "#f89e4f", "#2dc5f4", "#9253a1"];
        return pallet[index % pallet.length];
    }

    let props: Props = $props();
</script>

<div class="font-mono whitespace-pre-wrap leading-tight">
    {#each props.tokens as token, index}
        <span style="background-color: {getHue(index)};">{token}</span>
    {/each}
</div>

<script lang="ts">
	import type { Message } from '$lib/stores/chatStore.svelte';

	const { message }: { message: Message } = $props();

	function formatTime(timestamp: string): string {
		const date = new Date(timestamp);
		return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
	}
</script>

<div class={`mb-4 flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
	<div
		class={`max-w-3/4 rounded-lg p-3 ${
			message.role === 'user' ? 'bg-green-800 text-gray-100' : 'bg-gray-700 text-gray-100'
		}`}
	>
		<p class="whitespace-pre-wrap">{message.content}</p>
		{#if message.role === 'assistant' && message.meta}
			<div class="meta">
				<span>Model: {message.meta?.model}</span>
				<span>Time: {message.meta?.time ? Math.round(message.meta.time / 1e9) + 's' : ''}</span>
			</div>
		{/if}
	</div>
</div>

<style>
	.chat-message {
		margin-bottom: 0.5rem;
	}
	.chat-message.user .bubble {
		background: #222;
		color: #fff;
		align-self: flex-end;
	}
	.chat-message.assistant .bubble {
		background: #f3f3f3;
		color: #222;
		align-self: flex-start;
	}
	.bubble {
		border-radius: 1rem;
		padding: 0.75rem 1.25rem;
		max-width: 80%;
		display: inline-block;
	}
	.meta {
		font-size: 0.8em;
		color: #888;
		margin-top: 0.25em;
	}
</style>

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
		<div class="mt-1 text-right text-xs text-gray-400">
			{formatTime(message.timestamp)}
		</div>
	</div>
</div>

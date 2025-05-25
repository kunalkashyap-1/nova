<script lang="ts">
	import { page } from '$app/state';
	import { getContext } from 'svelte';
	import {
		chatState,
		loadConversations,
		loadConversation,
	} from '$lib/stores/chatStore.svelte';
	import ChatMessage from '$lib/ChatMessage.svelte';

	const CHAT_CONTAINER = Symbol('CHAT_CONTAINER');
	const chatContainerContext = getContext<{ set: (el: HTMLElement | null) => void } | undefined>(
		CHAT_CONTAINER
	);
	let chatContainer: HTMLElement | null = null;
	let userScrolledUp = $state(false);
	let chatId = $state(page.params.id);

	function handleScroll() {
		if (!chatContainer) return;
		userScrolledUp =
			chatContainer.scrollTop + chatContainer.clientHeight < chatContainer.scrollHeight - 40;
	}

	$effect(() => {
		chatId = page.params.id;
	});

	$effect(() => {
		loadConversations();
		// loadConversation(chatId);
	});

	$effect(() => {
		if (chatState.activeConversationId !== chatId) {
			// loadConversation(chatId);
		}
	});

	$effect(() => {
		if (chatContainer) {
			chatContainer.removeEventListener('scroll', handleScroll);
			chatContainer.addEventListener('scroll', handleScroll);
			return () => chatContainer && chatContainer.removeEventListener('scroll', handleScroll);
		}
	});

	$effect(() => {
		if (chatContainerContext && typeof chatContainerContext.set === 'function') {
			chatContainerContext.set(chatContainer);
		}
	});

    // Auto-scroll to bottom when new messages arrive or when streaming content updates
    $effect(() => {
        if (chatContainer && !userScrolledUp && chatState.messages.length > 0) {
            setTimeout(() => {
                if (chatContainer) {
                    chatContainer.scrollTo({
                        top: chatContainer.scrollHeight,
                        behavior: 'smooth'
                    });
                }
            }, 100);
        }
    });
</script>

<div class="mx-auto flex min-h-0 w-full flex-1 flex-col">
	<section class="max-h-full min-h-0 flex-1 overflow-y-auto px-13 chat-container" bind:this={chatContainer}>
		{#if chatState.errorMessage}
			<div class="mb-4 rounded border border-red-700 bg-red-900 px-4 py-3 text-red-100">
				{chatState.errorMessage}
			</div>
		{/if}

		{#if chatState.isLoading && chatState.messages.length === 0}
			<div class="flex justify-center py-8">
				<div class="animate-pulse text-gray-400">Loading messages...</div>
			</div>
		{:else}
			{#each chatState.messages as message (message.id)}
				<ChatMessage {message} />
			{/each}

			{#if chatState.isLoading && chatState.messages.length > 0}
				<div class="flex py-4">
					<div class="ml-12 h-6 w-12 animate-pulse rounded-lg bg-gray-700 p-3"></div>
				</div>
			{/if}
		{/if}
	</section>
</div>

<style>
	/* For Webkit browsers (Chrome, Safari) */
	.chat-container::-webkit-scrollbar {
		width: 8px;
	}

	.chat-container::-webkit-scrollbar-track {
		background: transparent;
	}

	.chat-container::-webkit-scrollbar-thumb {
		background: rgba(156, 163, 175, 0.5);
		border-radius: 4px;
	}

	.chat-container::-webkit-scrollbar-thumb:hover {
		background: rgba(156, 163, 175, 0.7);
	}

	/* For Firefox */
	.chat-container {
		scrollbar-width: thin;
		scrollbar-color: rgba(156, 163, 175, 0.5) transparent;
	}

	/* For Edge and IE */
	.chat-container {
		-ms-overflow-style: none;
	}
</style>

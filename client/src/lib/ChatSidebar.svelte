<script lang="ts">
	const { isSidebarCollapsed, toggleSidebar, openNewChatModal, activateVoiceInput } = $props();
	import { ChevronLeft, ChevronRight, Search, User, Mic } from '@lucide/svelte';
	import { chatState, loadConversations, signOut, authUser } from '$lib/stores/chatStore.svelte';
	import type { Conversation } from '$lib/stores/chatStore.svelte';
	import { goto } from '$app/navigation';

	let sortedConversations: Conversation[] = $derived(
		[...chatState.conversations].sort(
			(a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
		)
	);
</script>

{#if !isSidebarCollapsed}
	<aside
		class="relative z-10 flex h-full flex-col border-r border-gray-700 bg-gray-800 transition-all duration-300 ease-in-out"
		style="width: 16rem; min-width: 16rem; max-width: 16rem;"
	>
		<div class="flex items-center justify-between border-b border-gray-700 p-4">
			<button
				type="button"
				class="m-0 cursor-pointer border-0 bg-transparent p-0 text-xl font-bold text-green-400 focus:outline-none"
				onclick={() => goto('/chat')}
			>
				Nova
			</button>
			<button
				onclick={toggleSidebar}
				aria-label="Collapse sidebar"
				class="rounded p-1 hover:bg-gray-700"
			>
				<ChevronLeft class="h-5 w-5" />
			</button>
		</div>
		<div class="flex-1 overflow-y-auto p-4">
			<button
				onclick={openNewChatModal}
				class="mb-4 flex w-full items-center justify-center gap-2 rounded bg-green-600 py-2 text-sm text-white transition hover:bg-green-700"
			>
				<span>New Chat</span>
				<Mic class="h-4 w-4 cursor-pointer" onclick={activateVoiceInput} />
			</button>

			{#if chatState.isLoading && chatState.conversations.length === 0}
				<div class="py-4 text-center text-gray-400">Loading conversations...</div>
			{:else if chatState.conversations.length === 0}
				<div class="py-4 text-center text-gray-400">No conversations yet</div>
			{:else}
				<ul class="space-y-2">
					{#each sortedConversations as conversation (conversation.id)}
						<li
							class={`cursor-pointer rounded p-2 transition ${chatState.activeConversationId === conversation.id ? 'bg-gray-700' : 'hover:bg-gray-700'}`}
						>
							<button onclick={() => goto(`/chat/${conversation.id}`)} class="w-full">
								{conversation.title}
							</button>
						</li>
					{/each}
				</ul>
			{/if}
		</div>
		<div class="border-t border-gray-700 p-4">
			<div class="flex items-center justify-between rounded-lg bg-gray-700/50 p-3">
				<div class="flex items-center space-x-2">
					{#if authUser.profilePicUrl}
						<img src={authUser.profilePicUrl} alt="avatar" class="h-8 w-8 rounded-full" />
					{:else}
						<User class="h-8 w-8 text-gray-400" />
					{/if}
					<div>
						<div class="text-sm font-semibold">
							{authUser.isGuest ? 'Guest User' : authUser.username}
						</div>
						{#if !authUser.isGuest}
							<button onclick={signOut} class="text-xs text-green-400 hover:underline">
								Sign Out
							</button>
						{:else}
							<button onclick={() => goto('/')} class="text-xs text-green-400 hover:underline">
								Sign In
							</button>
						{/if}
					</div>
				</div>
			</div>
		</div>
	</aside>
{/if}

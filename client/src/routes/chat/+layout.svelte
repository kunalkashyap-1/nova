<script lang="ts">
	import ChatSidebar from '$lib/ChatSidebar.svelte';
	import InputBar from '$lib/InputBar.svelte';
	import Modal from '$lib/Modal.svelte';
	import { ChevronRight, Search } from '@lucide/svelte';
	import {
		chatState,
		sendMessage,
		setSelectedModel,
		createNewChat
	} from '$lib/stores/chatStore.svelte';
	import { goto } from '$app/navigation';

	let { children } = $props();
	let isSidebarCollapsed = $state(false);
	let showNewChatModal = $state(false);
	let newChatTitle = $state('');
	let modalTitle = $state('New Chat');
	let modalInputArea: HTMLTextAreaElement;

	function toggleSidebar() {
		isSidebarCollapsed = !isSidebarCollapsed;
	}

	function openNewChatModal() {
		showNewChatModal = true;
		newChatTitle = '';
	}

	function closeNewChatModal() {
		showNewChatModal = false;
		newChatTitle = '';
	}

	async function handleModalInput(e: Event) {
		const value = (e.target as HTMLTextAreaElement).value;
		newChatTitle = value;
		autoResizeModalInput();
	}

	function autoResizeModalInput() {
		if (modalInputArea) {
			modalInputArea.style.height = 'auto';
			const lines = modalInputArea.value.split('\n').length;
			const maxHeight = 7 * 24; // 7 lines, 24px per line
			const newHeight = Math.min(modalInputArea.scrollHeight, maxHeight);
			modalInputArea.style.height = `${newHeight}px`;
		}
	}

	function modalInputRef(node: HTMLTextAreaElement) {
		modalInputArea = node;
		setTimeout(() => node && node.focus(), 0);
	}

	async function handleNewChatModalSubmit() {
		if (!newChatTitle.trim()) return;
		const prompt = newChatTitle;
		const newId = await createNewChat(prompt);
		if (newId) {
			goto(`/chat/${newId}`);
		}
		closeNewChatModal();
	}

	function activateVoiceInput(event: Event) {
		event.stopPropagation();
		alert('Voice input functionality will be implemented later');
	}

	function handleSendMessage(content: string) {
		if (!chatState.activeConversationId) {
			createNewChat(content).then((newId) => {
				if (newId) {
					goto(`/chat/${newId}`);
				}
			});
		} else {
			sendMessage(content);
		}
	}
</script>

<div class="flex h-screen overflow-hidden bg-gray-900 text-gray-200">
	<!-- Sidebar -->
	<div class="relative">
		<div
			class={`transition-all duration-500 ease-in-out ${isSidebarCollapsed ? 'pointer-events-none w-0 opacity-0' : 'w-64 opacity-100'}`}
			style="min-width: 0; height: 100%;"
		>
			<ChatSidebar {isSidebarCollapsed} {toggleSidebar} {openNewChatModal} {activateVoiceInput} />
		</div>
		{#if isSidebarCollapsed}
			<div
				class="animate-fade-in fixed left-4 top-4 z-30 flex cursor-pointer items-center gap-2 rounded-xl border border-gray-700 bg-gray-800 p-2 shadow-lg transition-all duration-500"
				style=""
				role="region"
				aria-label="Collapsed sidebar controls"
			>
				<button
					class="rounded p-2 hover:bg-gray-700"
					aria-label="Expand sidebar"
					onclick={toggleSidebar}
				>
					<ChevronRight class="h-6 w-6 text-green-400" />
				</button>
				<button
					class="rounded p-2 hover:bg-gray-700"
					aria-label="New Chat"
					onclick={openNewChatModal}
				>
					<Search class="h-6 w-6 text-green-400" />
				</button>
			</div>
		{/if}
	</div>
	<!-- Main content area -->
	<div class="flex min-w-0 flex-1 flex-col transition-all duration-500 ease-in-out">
		<div class="flex min-h-0 flex-1 flex-col">
			<!-- Centered container for chat and input bar -->
			<div class="mx-auto w-full max-w-4xl flex flex-col min-h-0 flex-1 h-full">
				{@render children()}
				<div
					class="animate-fade-in sticky bottom-0 left-0 right-0 z-10 bg-gray-900/95 shadow-xl transition-all duration-500"
				>
					<InputBar
						onSendMessage={handleSendMessage}
						isLoading={chatState.isLoading}
						disabled={chatState.isLoading}
						selectedModel={chatState.selectedModel}
						onModelChange={setSelectedModel}
						onVoiceInput={activateVoiceInput}
					/>
				</div>
			</div>
		</div>
	</div>
	<!-- New Chat Modal -->
	{#if showNewChatModal}
		<Modal onclose={closeNewChatModal}>
			<div class="p-4">
				<div class="mb-2 text-lg font-semibold text-green-400">{modalTitle}</div>
				<textarea
					bind:value={newChatTitle}
					use:modalInputRef
					rows="1"
					maxlength="500"
					placeholder="Ask Anything..."
					class="max-h-[168px] min-h-[24px] w-full resize-none rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-gray-200 focus:outline-none focus:ring-2 focus:ring-green-500"
					oninput={handleModalInput}
					onkeydown={(e) => {
						if (e.key === 'Enter' && !e.shiftKey) {
							e.preventDefault();
							handleNewChatModalSubmit();
						}
					}}
				></textarea>
				{#if !newChatTitle.trim()}
					<div class="mt-1 text-xs text-gray-400">
						Press Enter to send, Shift+Enter for new line
					</div>
				{/if}
			</div>
		</Modal>
	{/if}
</div>

<style>
	@keyframes fade-in {
		0% {
			opacity: 0;
			transform: translateY(20px);
		}
		100% {
			opacity: 1;
			transform: translateY(0);
		}
	}
	.animate-fade-in {
		animation: fade-in 0.5s;
	}
</style>

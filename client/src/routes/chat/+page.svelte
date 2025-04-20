<script lang="ts">
	import { goto } from '$app/navigation';
	import {
		loadConversations,
		createNewChat,
		setError,
	} from '$lib/stores/chatStore.svelte';
	import Modal from '$lib/Modal.svelte';

	let showNewChatModal = $state(false);
	let newChatTitle = $state('');
	let modalTitle = $state('New Chat');
	let modalInputArea: HTMLTextAreaElement;

	$effect(() => {
		(async () => {
			try {
				await loadConversations();
			} catch (error) {
				setError('Failed to load conversations');
			}
		})();
	});


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

</script>

<div class="flex flex-1 flex-col min-h-0">
	<!-- this will have some temp how can i help and faq stuff -->
</div>

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
        class="custom-scrollbar max-h-[168px] min-h-[24px] w-full resize-none rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-gray-200 focus:outline-none focus:ring-2 focus:ring-green-500"
        oninput={handleModalInput}
        onkeydown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey && newChatTitle.trim()) {
            e.preventDefault();
            handleNewChatModalSubmit();
          }
        }}
      ></textarea>
      {#if !newChatTitle.trim()}
        <div class="mt-1 text-xs text-gray-400">Press Enter to send, Shift+Enter for new line</div>
      {/if}
    </div>
  </Modal>
{/if}

<style>
	.custom-scrollbar {
		scrollbar-width: thin;
		scrollbar-color: #22c55e transparent;
	}
	.custom-scrollbar::-webkit-scrollbar {
		width: 6px;
		background: transparent;
	}
	.custom-scrollbar::-webkit-scrollbar-thumb {
		background: #22c55e;
		border-radius: 8px;
	}
	.custom-scrollbar::-webkit-scrollbar-track {
		background: transparent;
	}
</style>

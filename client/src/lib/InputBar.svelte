<script lang="ts">
	import { Send, Mic, Search } from '@lucide/svelte';
	import ModelSelector from './components/ModelSelector.svelte';
	import { selectedModel } from './stores/modelStore.svelte';

	interface Props {
		onSendMessage: (content: string) => void;
		onVoiceInput: (event: Event) => void;
		isLoading?: boolean;
		disabled?: boolean;
	}

	let {
		onSendMessage,
		onVoiceInput,
		isLoading = false,
		disabled = false
	}: Props = $props();
	let message = $state('');
	let textareaElement: HTMLTextAreaElement;
	let autoFocused = false;

	function handleSend() {
		if (message.trim() && !isLoading && !disabled) {
			onSendMessage(message);
			message = '';
			if (textareaElement) {
				textareaElement.style.height = 'auto';
			}
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSend();
		}
	}

	function autoResize() {
		if (textareaElement) {
			textareaElement.style.height = 'auto';
			const newHeight = Math.min(textareaElement.scrollHeight, 150);
			textareaElement.style.height = `${newHeight}px`;
		}
	}

	$effect(() => {
		if (textareaElement) autoResize();
		if (textareaElement && !autoFocused) {
			textareaElement.focus();
			autoFocused = true;
		}
	});
</script>

<div
	class="
    flex
    w-full
    items-center
    justify-center
  "
>
	<div
		class="
      w-full
      max-w-[50rem]
      rounded-t-xl
      border-t
      border-gray-700
      bg-gray-800
      p-2
    "
	>
		<div
			class="
        flex
        flex-col
        items-start
        justify-center
        gap-2
      "
		>
			<div
				class="
          flex
          w-full
          flex-grow
          items-center
          rounded-lg
          bg-gray-700
          p-2
        "
			>
				<button
					onclick={onVoiceInput}
					class="
            p-2
            text-gray-400
            hover:text-green-400
            sm:p-1
          "
					aria-label="Voice input"
					{disabled}
				>
					<Mic class="h-5 w-5 sm:h-5 sm:w-5" />
				</button>

				<textarea
					bind:value={message}
					bind:this={textareaElement}
					oninput={autoResize}
					onkeydown={handleKeydown}
					placeholder={disabled ? 'Select or create a conversation' : 'Type a message...'}
					rows="1"
					class="
            max-h-32
            flex-grow
            resize-none
            border-0
            bg-transparent
            px-2
            py-2
            text-base
            text-gray-200
            outline-none
            focus:ring-0
          "
					{disabled}
				></textarea>

				<button
					onclick={handleSend}
					class={`
            rounded-full
            p-2
            sm:p-1
            ${
							!message.trim() || isLoading || disabled
								? 'cursor-not-allowed text-gray-500'
								: 'text-green-400 hover:text-green-300'
						}
          `}
					disabled={!message.trim() || isLoading || disabled}
					aria-label="Send message"
				>
					<Send class="h-5 w-5 sm:h-5 sm:w-5" />
				</button>
			</div>

			<div
				class="
          flex
          flex-row
          gap-2
        "
			>
				<ModelSelector />
				<button
					type="button"
					class="
            flex
            w-auto
            items-center
            justify-center
            gap-1
            rounded-full
            border-0
            bg-gray-700/50
            px-2
            py-0.5
            text-base
            text-gray-400
            transition-colors
            duration-150
            hover:bg-gray-600/50
            hover:text-green-400
            focus:outline-none
            sm:w-full
            sm:px-1
            sm:py-0.5
            sm:text-sm
          "
					aria-label="Search"
				>
					<Search class="h-5 w-5 sm:h-4 sm:w-4" />
					<span>Search</span>
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	/* Custom thin scrollbar for textarea */
	textarea.flex-grow::-webkit-scrollbar {
		width: 6px;
	}
	textarea.flex-grow::-webkit-scrollbar-thumb {
		background: #374151;
		border-radius: 8px;
	}
	textarea.flex-grow::-webkit-scrollbar-track {
		background: #111827;
	}
	textarea.flex-grow {
		scrollbar-width: thin;
		scrollbar-color: #374151 #111827;
	}
</style>

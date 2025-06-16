<script lang="ts">
	import { ChevronDown, Settings, Info } from 'lucide-svelte';
	import { modelStore, selectedModel, modelParameters } from '../stores/modelStore.svelte';
	import { onMount } from 'svelte';
	import type { Model } from '$lib/types';
	import { Dropdown, DropdownItem, Button, Spinner, Tooltip } from 'flowbite-svelte';
	import { fade } from 'svelte/transition';

	// State
	let showSettings = $state(false);
	let openDropdown = $state(false);

	// Handle model selection
	function handleModelSelect(modelId: number) {
		modelStore.selectModel(modelId);
		openDropdown = false;
	}

	// Truncate description to specified length
	function truncateDescription(description: string, maxLength: number = 60): string {
		if (!description || description.length <= maxLength) return description;
		return description.substring(0, maxLength).trim() + '...';
	}

	// Handle parameter update
	// function handleParameterUpdate(name: string, value: number) {
	// 	modelStore.updateParameter(name, value);
	// }

	onMount(() => {
		// Initial fetch of models
		modelStore.fetchModels();
	});
</script>

<div class="relative">
	<Button
		class="h-10 max-w-[200px] overflow-hidden truncate whitespace-nowrap border-0 bg-gray-700 text-gray-200 hover:bg-gray-600"
		color="alternative"
		onclick={() => (openDropdown = !openDropdown)}
	>
		{#if $modelStore.loading}
			<Spinner size="4" class="mr-2" />
			Loading...
		{:else if $modelStore.error}
			<span class="text-red-500">Error</span>
		{:else if $selectedModel}
			{$selectedModel.name}
		{:else}
			Select Model
		{/if}
	</Button>

	<Dropdown
		simple
		transition={fade}
		transitionParams={{ duration: 300 }}
		class="bg-gray-800"
		isOpen={openDropdown}
	>
		<!-- Loading State -->
		{#if $modelStore.loading}
			<div class="flex items-center justify-center p-8">
				<Spinner size="8" />
			</div>
			<!-- Error State -->
		{:else if $modelStore.error}
			<div class="flex flex-col items-center justify-center p-8 text-center">
				<span class="mb-2 text-red-500">{$modelStore.error}</span>
				<Button color="alternative" onclick={() => modelStore.fetchModels()}>Retry</Button>
			</div>
			<!-- Model List -->
		{:else}
			<div class="max-h-[32rem] overflow-y-auto bg-gray-800 p-4">
				<div class="flex w-full flex-col gap-2">
					{#each $modelStore.models as model}
						<DropdownItem
							class="group relative flex w-full flex-col rounded-lg border 
							border-gray-700 bg-gray-800 p-3 transition-all hover:border-green-500/50 
							hover:bg-gray-700 {$selectedModel?.id === model.id ? 'border-green-500 bg-gray-700' : ''}"
							onclick={() => handleModelSelect(model.id)}
						>
							<div class="flex w-full items-center justify-between">
								<div class="flex-1">
									<h3 class="font-medium text-gray-200">{model.name}</h3>
									{#if model.description}
										<div class="mt-1 flex items-center gap-2">
											<p class="text-sm text-gray-400">
												{truncateDescription(model.description)}
											</p>
											{#if model.description.length > 60}
												<div class="relative">
													<Info 
														id="info-{model.id}" 
														class="h-4 w-4 text-gray-500 hover:text-gray-300 cursor-help flex-shrink-0" 
													/>
													<Tooltip 
														triggeredBy="#info-{model.id}" 
														class="bg-gray-900 text-gray-200 border border-gray-600 max-w-xs text-sm"
														placement="top"
													>
														{model.description}
													</Tooltip>
												</div>
											{/if}
										</div>
									{/if}
								</div>
							</div>
							<div class="mt-2 flex flex-wrap gap-2">
								{#if model.cost_per_1k_input_tokens}
									<span class="rounded-full bg-green-900/50 px-2 py-0.5 text-xs text-green-400">
										${model.cost_per_1k_input_tokens}/1K tokens
									</span>
								{/if}
								{#if model.max_tokens}
									<span class="rounded-full bg-purple-900/50 px-2 py-0.5 text-xs text-purple-400">
										{model.max_tokens} max tokens
									</span>
								{/if}
							</div>
						</DropdownItem>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Model Settings -->
		<!-- {#if showSettings && $selectedModel}
			<hr class="my-2 border-gray-700" />
			<div class="p-3 bg-gray-800">
				<div class="mb-2 flex items-center justify-between">
					<h3 class="text-sm font-medium text-gray-200">Model Parameters</h3>
					<Button
						color="alternative"
						size="xs"
						class="h-8 w-8 bg-gray-700 hover:bg-gray-600"
						onclick={() => {
							showSettings = false;
						}}
					>
						<ChevronDown class="h-4 w-4" />
					</Button>
				</div>
				<div class="space-y-3">
					<div>
						<label for="temperature" class="mb-1 block text-sm text-gray-400">Temperature</label>
						<input
							type="range"
							id="temperature"
							min={0}
							max={1}
							step={0.1}
							value={$modelParameters.temperature}
							oninput={(e) =>
								handleParameterUpdate('temperature', parseFloat(e.currentTarget.value))}
							class="h-2 w-full cursor-pointer appearance-none rounded-lg bg-gray-700"
						/>
						<div class="mt-1 flex justify-between text-xs text-gray-400">
							<span>0.0</span>
							<span>{$modelParameters.temperature}</span>
							<span>1.0</span>
						</div>
					</div>
					<div>
						<label for="top-p" class="mb-1 block text-sm text-gray-400">Top P</label>
						<input
							type="range"
							id="top-p"
							min={0}
							max={1}
							step={0.1}
							value={$modelParameters.top_p}
							oninput={(e) => handleParameterUpdate('top_p', parseFloat(e.currentTarget.value))}
							class="h-2 w-full cursor-pointer appearance-none rounded-lg bg-gray-700"
						/>
						<div class="mt-1 flex justify-between text-xs text-gray-400">
							<span>0.0</span>
							<span>{$modelParameters.top_p}</span>
							<span>1.0</span>
						</div>
					</div>
					<div>
						<label for="max-tokens" class="mb-1 block text-sm text-gray-400">Max Tokens</label>
						<input
							type="number"
							id="max-tokens"
							min={1}
							max={4096}
							value={$modelParameters.max_tokens}
							oninput={(e: any) =>
								handleParameterUpdate('max_tokens', parseInt(e.currentTarget.value))}
							class="w-full rounded-lg bg-gray-700 px-3 py-2 text-gray-200 focus:border-green-500 focus:ring-green-500"
						/>
					</div>
				</div>
			</div>
		{/if} -->
	</Dropdown>
</div>

<style>
	/* Custom scrollbar */
	.overflow-y-auto::-webkit-scrollbar {
		width: 6px;
	}

	.overflow-y-auto::-webkit-scrollbar-track {
		background: #1f2937;
	}

	.overflow-y-auto::-webkit-scrollbar-thumb {
		background: #374151;
		border-radius: 3px;
	}

	.overflow-y-auto::-webkit-scrollbar-thumb:hover {
		background: #4b5563;
	}

	/* Range input styling */
	input[type='range']::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: #10b981;
		cursor: pointer;
	}

	input[type='range']::-moz-range-thumb {
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: #10b981;
		cursor: pointer;
		border: none;
	}
</style>
<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { authStore } from '$lib/stores/auth.svelte';
	let { children } = $props();

	// Ensure session persists by verifying auth status on page load
	onMount(() => {
		authStore.checkAuth();
	});

</script>

<div class="flex h-screen bg-gray-900 text-gray-200 overflow-hidden">
	<div class="flex-1 flex-col min-w-0 transition-all duration-500 ease-in-out">
		<div class="flex-1 flex flex-col min-h-0">
			{@render children()}
		</div>
	</div>
</div>

<style>
</style>

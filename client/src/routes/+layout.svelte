<script lang="ts">
	import '../app.css';
	let { children } = $props();

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

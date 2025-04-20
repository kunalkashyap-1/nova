<script lang="ts">
    import { fade } from 'svelte/transition';
  
    const { onclose, children } = $props();
  
    function handleKeydown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        closeModal();
      }
    }
  
    function closeModal() {
      if (onclose) onclose();
    }
  
    $effect(() => {
      window.addEventListener('keydown', handleKeydown);
      document.body.style.overflow = 'hidden';
  
      return () => {
        window.removeEventListener('keydown', handleKeydown);
        document.body.style.overflow = '';
      };
    });
  
    function handleBackdropClick(event: MouseEvent) {
      if (event.target === event.currentTarget) {
        closeModal();
      }
    }
  
    function handleBackdropKeydown(event: KeyboardEvent) {
      if (event.key === 'Enter') {
        closeModal();
      }
    }
  
  </script>
  
  <div
    class="fixed inset-0 bg-black/70 pt-42 flex items-start justify-center z-50 p-4"
    role="dialog"
    aria-modal="true"
    tabindex="0"
    onclick={handleBackdropClick}
    onkeydown={handleBackdropKeydown}
    transition:fade={{ duration: 200 }}
  >
    <div
      class="bg-gray-800 rounded-xl shadow-xl w-full max-w-lg relative"
      transition:fade={{ duration: 150 }}
    >
      {@render children()}
    </div>
  </div>
  
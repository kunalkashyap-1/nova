<script lang="ts">
    const { isSidebarCollapsed, toggleSidebar, openNewChatModal, activateVoiceInput } = $props();
    import { ChevronLeft, ChevronRight, Search, User, Mic } from '@lucide/svelte';
    import { chatState, loadConversations } from '$lib/stores/chatStore.svelte';
    import { onMount } from 'svelte';
    import { authStore } from '$lib/stores/auth.svelte';
    import type { Conversation } from '$lib/stores/chatStore.svelte';
    let sortedConversations: Conversation[] = $derived(
        [...chatState.conversations].sort(
            (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
        )
    );
    import { goto } from '$app/navigation';
    
    
        

    // Handle sign out
    async function handleSignOut() {
        try {
            await authStore.logout();
            goto('/');
        } catch (error) {
            console.error('Sign out error:', error);
        }
    }
    onMount(() => {
        loadConversations();
    });
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
                            <button onclick={() => goto(`/chat/${conversation.id}`)} class="w-full text-left">
                                <div class="truncate text-sm text-gray-200">
                                    {conversation.title}
                                </div>
                            </button>
                        </li>
                    {/each}
                </ul>
            {/if}
        </div>
        
        <div class="border-t border-gray-700 p-4">
            <div class="flex items-center justify-between rounded-lg bg-gray-700/50 p-3">
                <div class="flex items-center space-x-3">
                    {#if authStore.user?.profile_picture}
                        <img 
                            src={`http://localhost:8000/api/v1/auth/media/${authStore.user.profile_picture}`} 
                            alt="Profile picture" 
                            class="h-8 w-8 rounded-full object-cover" 
                        />
                    {:else}
                        <div class="flex h-8 w-8 items-center justify-center rounded-full bg-gray-600">
                            <User class="h-5 w-5 text-gray-300" />
                        </div>
                    {/if}
                    
                    <div class="flex-1 min-w-0">
                        {#if authStore.isAuthenticated && authStore.user}
                            <div class="text-sm font-medium text-gray-200 truncate">
                                {authStore.user.full_name || authStore.user.username}
                            </div>
                            <button 
                                onclick={handleSignOut} 
                                class="text-xs text-green-400 hover:text-green-300 hover:underline transition-colors"
                            >
                                Sign Out
                            </button>
                        {:else}
                            <div class="text-sm font-medium text-gray-200">
                                Guest User
                            </div>
                            <button 
                                onclick={() => goto('/')} 
                                class="text-xs text-green-400 hover:text-green-300 hover:underline transition-colors"
                            >
                                Sign In
                            </button>
                        {/if}
                    </div>
                </div>
                
                <!-- {#if authStore.isLoading}
                    <div class="h-2 w-2 animate-pulse rounded-full bg-gray-400"></div>
                {/if} -->
            </div>
        </div>
    </aside>
{/if}
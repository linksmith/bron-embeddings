<script>
    import { createEventDispatcher } from 'svelte';
    import Citation from './Citation.svelte';
  
    export let messages = [];
  
    let inputMessage = '';
    const dispatch = createEventDispatcher();
  
    function sendMessage() {
      if (inputMessage.trim()) {
        dispatch('newMessage', { role: 'user', content: inputMessage });
        inputMessage = '';
      }
    }
  
    function handleCitationClick(id) {
      dispatch('citationClick', id);
    }
  </script>
  
  <div class="flex flex-col h-full bg-white rounded-lg shadow-md">
    <div class="flex-1 overflow-y-auto p-4 space-y-4">
      {#each messages as message}
        <div class="flex flex-col {message.role === 'user' ? 'items-end' : 'items-start'}">
          <div class="max-w-3/4 p-2 rounded-lg {message.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'}">
            {#if message.role === 'assistant'}
              {#each message.content.split(/(\[[0-9]+\])/) as part}
                {#if part.match(/\[[0-9]+\]/)}
                  <Citation id={part.slice(1, -1)} on:click={() => handleCitationClick(part.slice(1, -1))} />
                {:else}
                  {part}
                {/if}
              {/each}
            {:else}
              {message.content}
            {/if}
          </div>
        </div>
      {/each}
    </div>
    <div class="p-4 border-t">
      <form on:submit|preventDefault={sendMessage} class="flex">
        <input
          type="text"
          bind:value={inputMessage}
          placeholder="Type your message..."
          class="flex-1 p-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button type="submit" class="px-4 py-2 bg-blue-500 text-white rounded-r-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500">
          Send
        </button>
      </form>
    </div>
  </div>
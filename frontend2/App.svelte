<script>
    import { onMount } from 'svelte';
    import Chat from './components/Chat.svelte';
    import Documents from './components/Documents.svelte';
  
    let messages = [];
    let documents = [];
    let selectedDocument = null;
  
    function handleNewMessage(event) {
      messages = [...messages, event.detail];
    }
  
    function handleNewDocuments(event) {
      documents = event.detail;
    }
  
    function handleCitationClick(event) {
      selectedDocument = documents.find(doc => doc.id === event.detail);
    }
  
    onMount(() => {
      // Initialize WebSocket connection
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.text) {
          handleNewMessage({ detail: { role: 'assistant', content: data.text, citations: data.citations } });
        }
        if (data.documents) {
          handleNewDocuments({ detail: data.documents });
        }
      };
    });
  </script>
  
  <main class="flex h-screen bg-gray-100">
    <div class="w-1/2 p-4">
      <Chat {messages} on:newMessage={handleNewMessage} on:citationClick={handleCitationClick} />
    </div>
    <div class="w-1/2 p-4">
      <Documents {documents} {selectedDocument} />
    </div>
  </main>
  
  <style global lang="postcss">
    @tailwind base;
    @tailwind components;
    @tailwind utilities;
  </style>
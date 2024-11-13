<script>
    export let documents = [];
    export let selectedDocument = null;
  </script>
  
  <div class="h-full bg-white rounded-lg shadow-md overflow-hidden">
    {#if selectedDocument}
      <div class="p-4">
        <h2 class="text-xl font-bold mb-2">{selectedDocument.title}</h2>
        <p class="text-sm text-gray-600 mb-4">ID: {selectedDocument.id}</p>
        <div class="prose max-w-none">
          {@html selectedDocument.content}
        </div>
      </div>
    {:else}
      <div class="p-4">
        <h2 class="text-xl font-bold mb-4">Related Documents</h2>
        {#if documents.length > 0}
          <ul class="space-y-2">
            {#each documents as doc}
              <li class="p-2 bg-gray-100 rounded">
                <h3 class="font-semibold">{doc.title}</h3>
                <p class="text-sm text-gray-600">ID: {doc.id}</p>
              </li>
            {/each}
          </ul>
        {:else}
          <p>No related documents found.</p>
        {/if}
      </div>
    {/if}
  </div>
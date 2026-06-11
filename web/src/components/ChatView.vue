<script setup>
import { ref, inject, onMounted, computed, nextTick } from 'vue'
import { marked } from 'marked'
import { api } from '../api.js'

const status = inject('status')
const approaches = ref({ order: [], approaches: {} })
const approach = ref('')
const query = ref('')
const messages = ref([])
const busy = ref(false)
const error = ref('')
const messagesEl = ref(null)

const indexReady = computed(() => !!status.value?.index)
const currentDesc = computed(() => approaches.value.approaches[approach.value]?.description || '')

onMounted(async () => {
  approaches.value = await api.approaches()
  approach.value = approaches.value.order[0] || ''
})

function md(t) { return marked.parse(t || '') }

async function send() {
  const q = query.value.trim()
  if (!q || busy.value || !approach.value) return
  error.value = ''
  messages.value.push({ role: 'user', text: q })
  query.value = ''
  busy.value = true
  await scroll()
  try {
    const res = await api.chat(approach.value, q)
    messages.value.push({ role: 'assistant', ...res })
  } catch (e) {
    error.value = e.message
    messages.value.push({ role: 'assistant', answer: '', error: e.message, contexts: [], trace: [] })
  } finally {
    busy.value = false
    await scroll()
  }
}

async function scroll() {
  await nextTick()
  if (messagesEl.value) messagesEl.value.scrollTop = messagesEl.value.scrollHeight
}
</script>

<template>
  <div class="chat-wrap">
    <div class="card" style="margin-bottom:12px">
      <div class="row" style="justify-content:space-between">
        <div class="row">
          <b>Approach&nbsp;</b>
          <select v-model="approach">
            <option v-for="k in approaches.order" :key="k" :value="k">{{ approaches.approaches[k]?.label }}</option>
          </select>
          <span class="muted" style="max-width:620px">{{ currentDesc }}</span>
        </div>
        <span class="pill" v-if="status?.models">retriever embed: {{ status.models.embed }}</span>
      </div>
    </div>

    <div v-if="!indexReady" class="banner warn">No index yet. Go to <b>Index &amp; Eval</b> and re-embed the Documents folder first.</div>
    <div v-if="error" class="banner err">{{ error }}</div>

    <div class="messages" ref="messagesEl">
      <div v-if="!messages.length" class="muted" style="padding:20px">
        Ask a question about the documents. The selected approach retrieves chunks, which are shown
        below the answer — always quoted verbatim with their source citation.
      </div>

      <div v-for="(m, i) in messages" :key="i" class="msg">
        <div class="who">{{ m.role === 'user' ? 'You' : 'Assistant · ' + (approaches.approaches[m.approach]?.label || approach) }}</div>

        <div v-if="m.role === 'user'" class="bubble user">{{ m.text }}</div>

        <div v-else class="bubble">
          <div v-if="m.error" class="banner err">{{ m.error }}</div>
          <div class="answer" v-html="md(m.answer)"></div>

          <div v-if="m.contexts?.length">
            <h3 style="margin-top:14px">📑 Retrieved &amp; quoted sources ({{ m.contexts.length }})</h3>
            <div class="chunk" v-for="(c, ci) in m.contexts" :key="ci">
              <div class="cite">
                <b>[{{ ci + 1 }}] {{ c.citation }}</b>
                <span class="score">score {{ c.score }}</span>
                <span class="tag">{{ c.stage }}</span>
              </div>
              <blockquote>{{ c.text }}</blockquote>
            </div>
          </div>

          <details v-if="m.trace?.length">
            <summary>🔎 Retrieval trace ({{ m.trace.length }} steps) · {{ m.latency_s }}s</summary>
            <div class="trace-step" v-for="(t, ti) in m.trace" :key="ti">
              <b>{{ t.label }}</b><span v-if="t.detail"> — {{ t.detail }}</span>
            </div>
          </details>
        </div>
      </div>
    </div>

    <div class="composer">
      <input type="text" v-model="query" placeholder="Ask about the documents…" @keyup.enter="send" :disabled="busy || !indexReady" />
      <button class="primary" @click="send" :disabled="busy || !indexReady || !query.trim()">
        <span v-if="busy" class="loader"></span><span v-else>Send</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, inject, onMounted, onUnmounted, computed } from 'vue'
import { api } from '../api.js'

const status = inject('status')
const refreshStatus = inject('refreshStatus')

const job = ref(null)
const error = ref('')
const rebuildGraph = ref(true)
const numGoldens = ref(16)
let timer = null

const busy = computed(() => job.value && job.value.status === 'running')

async function poll() {
  try {
    const j = await api.job()
    job.value = j.status === 'idle' ? null : j
    if (job.value && job.value.status !== 'running') {
      await refreshStatus()
      if (timer) { clearInterval(timer); timer = null }
    }
  } catch (e) { /* ignore transient */ }
}

function startPolling() {
  if (timer) clearInterval(timer)
  timer = setInterval(poll, 1200)
}

async function guard(fn) {
  error.value = ''
  try { await fn(); startPolling() } catch (e) { error.value = e.message }
}

const reembed = () => guard(() => api.reembed(rebuildGraph.value))
const synth = () => guard(() => api.synth(numGoldens.value))
const runEval = () => guard(() => api.evalRun(null))
const runFull = () => guard(() => api.full(numGoldens.value))

onMounted(() => { poll(); startPolling() })
onUnmounted(() => { if (timer) clearInterval(timer) })
</script>

<template>
  <div>
    <div v-if="error" class="banner err">{{ error }}</div>

    <div class="card">
      <h2>Pipeline control</h2>
      <h3>Embed the Documents folder, build the knowledge graph, synthesize goldens, and run DeepEval.</h3>
      <div class="row" style="margin-bottom:10px">
        <label class="row" style="gap:6px"><input type="checkbox" v-model="rebuildGraph" /> rebuild GraphRAG graph</label>
        <button class="primary" :disabled="busy" @click="reembed">↻ Re-embed Documents</button>
        <span style="width:1px;height:24px;background:var(--border)"></span>
        <label class="row" style="gap:6px">goldens <input type="number" v-model.number="numGoldens" min="4" max="60" style="width:70px" /></label>
        <button :disabled="busy" @click="synth">Synthesize goldens</button>
        <button :disabled="busy" @click="runEval">Run DeepEval (all approaches)</button>
        <span style="width:1px;height:24px;background:var(--border)"></span>
        <button :disabled="busy" @click="runFull">⚡ Run full pipeline</button>
      </div>

      <div v-if="job">
        <div class="row" style="justify-content:space-between">
          <div><b>{{ job.kind }}</b> · {{ job.status }} <span v-if="busy" class="loader"></span></div>
          <div class="muted">{{ job.elapsed_s }}s · {{ job.stage }}</div>
        </div>
        <div class="progressbar" style="margin:8px 0" v-if="job.progress >= 0">
          <div :style="{ width: (job.progress * 100) + '%' }"></div>
        </div>
        <div class="progressbar" style="margin:8px 0" v-else-if="busy">
          <div style="width:100%;opacity:.4"></div>
        </div>
        <div v-if="job.error" class="banner err">{{ job.error }}</div>
        <div class="log">{{ (job.log || []).slice(-12).join('\n') }}</div>
      </div>
    </div>

    <div class="grid" style="grid-template-columns: 1fr 1fr">
      <div class="card">
        <h2>Base index</h2>
        <div v-if="status?.index" class="kv">
          <b>embedded at</b><span>{{ status.index.embedded_at }}</span>
          <b>embed model</b><span>{{ status.index.embed_model }} ({{ status.index.embed_dim }}d)</span>
          <b>chunks</b><span>{{ status.index.num_chunks }} from {{ status.index.num_docs }} docs</span>
          <b>chunk size</b><span>{{ status.index.chunk_size }} / overlap {{ status.index.chunk_overlap }}</span>
          <b>build time</b><span>{{ status.index.build_seconds }}s</span>
        </div>
        <div v-else class="muted">No index yet — click <b>Re-embed Documents</b>.</div>
      </div>

      <div class="card">
        <h2>GraphRAG graph</h2>
        <div v-if="status?.graph" class="kv">
          <b>built at</b><span>{{ status.graph.built_at }}</span>
          <b>entities</b><span>{{ status.graph.entities }}</span>
          <b>relationships</b><span>{{ status.graph.relationships }}</span>
          <b>communities</b><span>{{ status.graph.communities }}</span>
          <b>chunks processed</b><span>{{ status.graph.chunks_processed }}</span>
        </div>
        <div v-else class="muted">No graph yet.</div>
      </div>
    </div>

    <div class="card">
      <h2>Models</h2>
      <div v-if="status?.models" class="kv">
        <b>embedding</b><span>{{ status.models.embed }}</span>
        <b>generation</b><span>{{ status.models.gen }}</span>
        <b>judge (DeepEval)</b><span>{{ status.models.judge }}</span>
        <b>reranker</b><span>{{ status.models.reranker }}</span>
      </div>
    </div>

    <div class="card">
      <h2>Documents ({{ status?.documents?.length || 0 }})</h2>
      <div class="row">
        <span class="tag" v-for="d in status?.documents" :key="d">{{ d }}
          <span class="muted" v-if="status?.index?.per_doc">· {{ status.index.per_doc[d] || 0 }}</span>
        </span>
      </div>
    </div>
  </div>
</template>

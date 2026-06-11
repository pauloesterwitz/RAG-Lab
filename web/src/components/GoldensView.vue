<script setup>
import { ref, onMounted } from 'vue'
import { api } from '../api.js'

const data = ref(null)
const error = ref('')

onMounted(async () => {
  try { data.value = await api.goldens() } catch (e) { error.value = e.message }
})
</script>

<template>
  <div>
    <div class="row" style="justify-content:space-between; margin-bottom:8px">
      <h2 style="margin:0">Synthesized goldens (gold chunks → Q&amp;A)</h2>
      <span class="pill" v-if="data">{{ data.num_goldens }} goldens · judge {{ data.judge_model }}</span>
    </div>
    <div v-if="error" class="banner err">{{ error }}</div>
    <div v-if="data && !data.num_goldens" class="banner warn">No goldens yet. Synthesize them in <b>Index &amp; Eval</b>.</div>

    <div class="card" v-for="(g, i) in data?.goldens" :key="i">
      <div class="row" style="justify-content:space-between">
        <b>Q{{ i + 1 }}. {{ g.input }}</b>
        <span class="tag">{{ g.source_file }}</span>
      </div>
      <div style="margin:8px 0"><span class="muted">Expected answer:</span> {{ g.expected_output }}</div>
      <details>
        <summary>Gold context ({{ g.context?.length || 0 }} chunk{{ (g.context?.length||0) === 1 ? '' : 's' }})</summary>
        <div class="chunk" v-for="(c, ci) in g.context" :key="ci"><blockquote>{{ c }}</blockquote></div>
      </details>
    </div>
  </div>
</template>

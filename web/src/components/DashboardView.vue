<script setup>
import { ref, onMounted, computed } from 'vue'
import { api } from '../api.js'
import Chart from './Chart.vue'

const data = ref(null)
const error = ref('')
const drill = ref('')

const PALETTE = ['#6ea8fe', '#7ee0c0', '#f0b36b', '#c69bf0', '#f08a8a', '#88d8f0']

async function load() {
  error.value = ''
  try { data.value = await api.metrics() } catch (e) { error.value = e.message }
}
onMounted(load)

const order = computed(() => Object.keys(data.value?.approaches || {}))
const metricOrder = computed(() => data.value?.metric_order || [])

function color(i) { return PALETTE[i % PALETTE.length] }

// best value per metric column (higher is better)
function bestFor(metric) {
  let best = -1
  for (const a of order.value) {
    const v = data.value.approaches[a].metrics[metric]
    if (v != null && v > best) best = v
  }
  return best
}
function bestComposite() {
  let best = -1
  for (const a of order.value) {
    const v = data.value.approaches[a].composite
    if (v != null && v > best) best = v
  }
  return best
}
const winner = computed(() => {
  let w = null, best = -1
  for (const a of order.value) {
    const v = data.value.approaches[a].composite
    if (v != null && v > best) { best = v; w = a }
  }
  return w ? { key: w, label: data.value.approaches[w].label, score: best } : null
})

const radarData = computed(() => ({
  labels: metricOrder.value,
  datasets: order.value.map((a, i) => ({
    label: data.value.approaches[a].label,
    data: metricOrder.value.map(m => data.value.approaches[a].metrics[m] ?? 0),
    borderColor: color(i), backgroundColor: color(i) + '22', borderWidth: 2, pointRadius: 2,
  })),
}))
const radarOpts = {
  scales: { r: { min: 0, max: 1, ticks: { color: '#8b92a4', backdropColor: 'transparent' },
    grid: { color: '#2a2f3c' }, angleLines: { color: '#2a2f3c' }, pointLabels: { color: '#c5cad6', font: { size: 11 } } } },
}

const compositeBar = computed(() => ({
  labels: order.value.map(a => data.value.approaches[a].label),
  datasets: [{ label: 'Composite DeepEval score',
    data: order.value.map(a => data.value.approaches[a].composite ?? 0),
    backgroundColor: order.value.map((a, i) => color(i)) }],
}))

const metricBars = computed(() => ({
  labels: order.value.map(a => data.value.approaches[a].label),
  datasets: metricOrder.value.map((m, i) => ({
    label: m, data: order.value.map(a => data.value.approaches[a].metrics[m] ?? 0),
    backgroundColor: color(i),
  })),
}))
const barOpts = { scales: { y: { min: 0, max: 1, ticks: { color: '#8b92a4' }, grid: { color: '#2a2f3c' } },
  x: { ticks: { color: '#8b92a4' }, grid: { color: 'transparent' } } } }

const extraCols = [
  { key: 'gold_chunk_hit_rate', label: 'Gold-chunk hit', higher: true, fmt: v => (v*100).toFixed(0)+'%' },
  { key: 'gold_doc_hit_rate', label: 'Gold-doc hit', higher: true, fmt: v => (v*100).toFixed(0)+'%' },
  { key: 'avg_latency_s', label: 'Avg latency', higher: false, fmt: v => v+'s' },
  { key: 'avg_num_contexts', label: 'Avg #chunks', higher: null, fmt: v => v },
  { key: 'avg_context_chars', label: 'Avg ctx chars', higher: null, fmt: v => v },
]
function bestExtra(col) {
  if (col.higher === null) return null
  let best = col.higher ? -1 : 1e9
  for (const a of order.value) {
    const v = data.value.approaches[a].extra[col.key]
    if (v == null) continue
    if (col.higher ? v > best : v < best) best = v
  }
  return best
}

const drillCases = computed(() => drill.value ? (data.value.approaches[drill.value]?.cases || []) : [])
function goldenInput(i) { return data.value.goldens?.[i]?.input || '(question)' }
</script>

<template>
  <div>
    <div class="row" style="justify-content:space-between; margin-bottom:8px">
      <h2 style="margin:0">DeepEval dashboard</h2>
      <button @click="load">↻ Refresh</button>
    </div>
    <div v-if="error" class="banner err">{{ error }}</div>

    <div v-if="!data?.has_results" class="banner warn">
      No evaluation results yet. Go to <b>Index &amp; Eval</b> and run DeepEval.
    </div>

    <template v-else>
      <div class="card">
        <div class="row" style="justify-content:space-between">
          <div class="kv">
            <b>judge</b><span>{{ data.judge_model }}</span>
            <b>generator</b><span>{{ data.gen_model }}</span>
            <b>embedder</b><span>{{ data.embed_model }}</span>
            <b>reranker</b><span>{{ data.reranker }}</span>
            <b>goldens</b><span>{{ data.num_goldens }}</span>
            <b>generated</b><span>{{ data.generated_at }}</span>
          </div>
          <div v-if="winner" style="text-align:right">
            <div class="muted">Strongest approach (composite)</div>
            <div style="font-size:22px" class="best">🏆 {{ winner.label }}</div>
            <div class="muted">composite {{ winner.score }}</div>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>DeepEval metrics by approach</h2>
        <h3>Answer Relevancy · Faithfulness · Contextual Relevancy / Precision / Recall · G-Eval. Best per column highlighted.</h3>
        <table>
          <thead>
            <tr>
              <th>Approach</th>
              <th v-for="m in metricOrder" :key="m">{{ m }}</th>
              <th>Composite</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="a in order" :key="a">
              <td><b>{{ data.approaches[a].label }}</b></td>
              <td class="metric" v-for="m in metricOrder" :key="m"
                  :class="{ best: data.approaches[a].metrics[m] != null && data.approaches[a].metrics[m] === bestFor(m) }">
                {{ data.approaches[a].metrics[m] ?? '—' }}
              </td>
              <td class="metric" :class="{ best: data.approaches[a].composite === bestComposite() }">
                {{ data.approaches[a].composite ?? '—' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="grid" style="grid-template-columns: 1fr 1fr">
        <div class="card"><h2>Metric profile (radar)</h2>
          <div style="height:340px"><Chart type="radar" :data="radarData" :options="radarOpts" /></div>
        </div>
        <div class="card"><h2>Composite score</h2>
          <div style="height:340px"><Chart type="bar" :data="compositeBar" :options="barOpts" /></div>
        </div>
      </div>

      <div class="card"><h2>Per-metric comparison</h2>
        <div style="height:360px"><Chart type="bar" :data="metricBars" :options="barOpts" /></div>
      </div>

      <div class="card">
        <h2>Retrieval-level metrics (beyond DeepEval)</h2>
        <h3>Gold-chunk / gold-doc hit rate measure whether retrieval surfaced the exact source the golden was made from. Best highlighted.</h3>
        <table>
          <thead><tr><th>Approach</th><th v-for="c in extraCols" :key="c.key">{{ c.label }}</th><th>Eval time</th></tr></thead>
          <tbody>
            <tr v-for="a in order" :key="a">
              <td><b>{{ data.approaches[a].label }}</b></td>
              <td class="metric" v-for="c in extraCols" :key="c.key"
                  :class="{ best: bestExtra(c) != null && data.approaches[a].extra[c.key] === bestExtra(c) }">
                {{ c.fmt(data.approaches[a].extra[c.key]) }}
              </td>
              <td class="metric muted">{{ data.approaches[a].eval_seconds }}s</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="card">
        <h2>Drill down: per-question scores</h2>
        <div class="row" style="margin-bottom:10px">
          <select v-model="drill">
            <option value="">— pick an approach —</option>
            <option v-for="a in order" :key="a" :value="a">{{ data.approaches[a].label }}</option>
          </select>
        </div>
        <div v-if="drill">
          <details v-for="(c, i) in drillCases" :key="i">
            <summary>
              <b>Q{{ i + 1 }}.</b> {{ goldenInput(i).slice(0, 110) }}
              <span class="muted">· composite cells below</span>
            </summary>
            <div style="padding:8px 0 16px">
              <table style="margin-bottom:10px">
                <thead><tr><th v-for="m in metricOrder" :key="m">{{ m }}</th><th>gold-chunk</th><th>latency</th></tr></thead>
                <tbody><tr>
                  <td class="metric" v-for="m in metricOrder" :key="m">{{ c[m]?.score ?? '—' }}</td>
                  <td>{{ c.gold_chunk_hit ? '✓' : '✗' }}</td>
                  <td>{{ c.latency_s }}s</td>
                </tr></tbody>
              </table>
              <details><summary>Answer</summary><div class="bubble" style="margin-top:6px">{{ c.answer }}</div></details>
              <details><summary>Retrieved &amp; quoted chunks ({{ c.retrieved?.length || 0 }})</summary>
                <div class="chunk" v-for="(rc, ri) in c.retrieved" :key="ri">
                  <div class="cite"><b>[{{ ri+1 }}] {{ rc.citation }}</b><span class="score">score {{ rc.score }}</span></div>
                  <blockquote>{{ rc.text.slice(0, 500) }}<span v-if="rc.text.length>500">…</span></blockquote>
                </div>
              </details>
              <details><summary>Metric reasons</summary>
                <div v-for="m in metricOrder" :key="m" class="trace-step"><b>{{ m }}:</b> {{ c[m]?.reason }}</div>
              </details>
            </div>
          </details>
        </div>
      </div>
    </template>
  </div>
</template>

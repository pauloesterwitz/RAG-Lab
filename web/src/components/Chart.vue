<script setup>
import { ref, onMounted, watch, onBeforeUnmount } from 'vue'
import Chart from 'chart.js/auto'

const props = defineProps({ type: String, data: Object, options: Object })
const el = ref(null)
let chart = null

function render() {
  if (chart) { chart.destroy(); chart = null }
  if (!el.value) return
  chart = new Chart(el.value, {
    type: props.type,
    data: props.data,
    options: Object.assign(
      { responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#c5cad6' } } } },
      props.options || {}
    ),
  })
}

onMounted(render)
watch(() => [props.data, props.type], render, { deep: true })
onBeforeUnmount(() => { if (chart) chart.destroy() })
</script>

<template>
  <div style="position:relative; height:100%; width:100%"><canvas ref="el"></canvas></div>
</template>

<script setup>
import { ref, onMounted, provide } from 'vue'
import { api } from './api.js'
import ReembedView from './components/ReembedView.vue'
import ChatView from './components/ChatView.vue'
import DashboardView from './components/DashboardView.vue'
import GoldensView from './components/GoldensView.vue'

const tab = ref('embed')
const status = ref(null)
const error = ref('')

const tabs = [
  { id: 'embed', label: 'Index & Eval' },
  { id: 'chat', label: 'Chat / Explore' },
  { id: 'dashboard', label: 'DeepEval Dashboard' },
  { id: 'goldens', label: 'Goldens' },
]

async function refreshStatus() {
  try { status.value = await api.status() } catch (e) { error.value = e.message }
}
provide('refreshStatus', refreshStatus)
provide('status', status)

onMounted(refreshStatus)
</script>

<template>
  <div class="app">
    <div class="topbar">
      <div class="brand">RAG&nbsp;<span>Lab</span></div>
      <div class="nav">
        <button v-for="t in tabs" :key="t.id" :class="{ active: tab === t.id }" @click="tab = t.id">
          {{ t.label }}
        </button>
      </div>
      <div class="spacer"></div>
      <span class="pill" v-if="status?.index">{{ status.index.num_chunks }} chunks · {{ status.index.num_docs }} docs</span>
      <span class="pill" v-if="status?.models">embed: {{ status.models.embed }}</span>
      <span class="pill" v-if="status?.models">judge: {{ status.models.judge }}</span>
    </div>

    <div class="main">
      <div class="container">
        <div v-if="error" class="banner err">{{ error }}</div>
        <ReembedView v-show="tab === 'embed'" />
        <ChatView v-show="tab === 'chat'" />
        <DashboardView v-show="tab === 'dashboard'" />
        <GoldensView v-show="tab === 'goldens'" />
      </div>
    </div>
  </div>
</template>

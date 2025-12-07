<script setup>
import axios from 'axios'
import PulseLoader from 'vue-spinner/src/PulseLoader.vue'
import { ref, computed } from 'vue'

const query = ref("")
const answer = ref("")
const references = ref([])
const isLoading = ref(false)

const showAdvanced = ref(false)
const toggleAdvanced = () => (showAdvanced.value = !showAdvanced.value)

const placeholders = [
  "What does salamander symbolize?",
  "Poem with fire and water",
]

const categories = ref({
  ghazal: true,
  masnavi: true,
  program: true,
})

const modelCategories = [
  "fast",
  "expert",
]

const modelCategory = ref("fast")

const activeCategories = computed(() =>
  Object.keys(categories.value).filter(k => categories.value[k])
)

const currentPlaceholder = ref('')

let placeholderIndex = 0
let charIndex = 0
let typingInterval, deletingInterval, pauseTimeout

function typeEffect() {
  const text = placeholders[placeholderIndex]

  if (charIndex < text.length) {
    currentPlaceholder.value += text.charAt(charIndex)
    charIndex++
  } else {
    clearInterval(typingInterval)
    pauseTimeout = setTimeout(startDeleting, 12000) // pause before deleting
  }
}

function startTyping() {
  currentPlaceholder.value = ''
  charIndex = 0
  typingInterval = setInterval(typeEffect, 50)
}

function startDeleting() {
  deletingInterval = setInterval(() => {
    if (charIndex > 0) {
      currentPlaceholder.value = currentPlaceholder.value.slice(0, -1)
      charIndex--
    } else {
      clearInterval(deletingInterval)
      placeholderIndex = (placeholderIndex + 1) % placeholders.length
      startTyping()
    }
  }, 40)
}


startTyping()

async function sendRagQuery() {
  isLoading.value = true
  try {
    const response = await axios.post('http://localhost:3000/rag', {
      query: query.value,
      model_category: modelCategory.value,
      selected_types: activeCategories.value,
    });
    answer.value = response.data['answer'];
    references.value = response.data['references'];
  } catch (error) {
    console.error("Error sending RAG query:", error);
    answer.value = "Error fetching data";
  } finally {
    isLoading.value = false
  }
}
</script>

<template>
  <div class="home">
    <div class="title">
      <h1>آتشعر</h1>
    </div>
    <div class="query">
      <input class="query-input" v-model="query" @keydown.enter="sendRagQuery" :placeholder="currentPlaceholder" />
      <button class="query-button" @click="sendRagQuery">ASK</button>
    </div>
    <div class="settings-container">
      <!-- Collapsible Header -->
      <button class="collapsible-header" @click="toggleAdvanced">
        Advanced Settings
        <span>{{ showAdvanced ? "▲" : "▼" }}</span>
      </button>

      <!-- Collapsible Content -->
      <transition name="fade">
        <div v-if="showAdvanced" class="collapsible-content">
          <div class="settings-group">
            <h4>Categories</h4>
            <div class="checkbox-group">
              <label>
                <input type="checkbox" v-model="categories.ghazal" /> Ghazal
              </label>
              <label>
                <input type="checkbox" v-model="categories.masnavi" /> Masnavi
              </label>
              <label>
                <input type="checkbox" v-model="categories.program" /> Programs
              </label>
            </div>
          </div>
          <div class="settings-group">
            <h4>Model Selection</h4>
            <div class="button-select">
              <button
                v-for="model in modelCategories"
                :key="model"
                @click="modelCategory = model"
                :class="{ active: modelCategory === model }"
              >
                {{ model }}
              </button>
            </div>
          </div>
        </div>
      </transition>
    </div>    
    <div v-if="isLoading" class="loader">
      <PulseLoader color="#25bcdf" />
    </div>
    <div v-else>
      <pre class="answer-text">{{ answer }}</pre>
      <ul class="answer-references">
        <li v-for="reference in references" class="answer-reference" :class="[reference.metadata.type]">
          <div class="reference-header" :class="[reference.metadata.type]">
            <span>{{ reference.metadata.type }}: {{ reference.metadata.number }}</span>
          </div>
          <div class="reference-content">
            <span class="reference-persian">{{ reference.page_content }}</span>
            <span class="reference-translation">{{ reference.metadata.translation }}</span>
          </div>
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.home {
  display: flex;
  justify-content: center;
  flex-flow: column wrap;
  align-content: stretch;
  max-width: 800px;
  margin: 0 auto;
  font-family: "Scheherazade New";
  line-height: 1.6em;

  .query {
    display: flex;
    
    .query-input {
      font-size: 1.4em;
      height: 40px;
      flex-grow: 1;
      padding: 10px;
      border-radius: 10px;
      transition: border-color 0.2s;
    }

    .query-button {
      font-size: 1.4em;
      width: 120px;
      margin-left: 5px;
      background-color: rgba(110, 218, 242, 1);
      border: 4px solid rgb(110, 189, 242, 1);
      border-radius: 10px;
      font-weight: bold;

      &:hover {
        background-color: rgb(110, 189, 242, 1);
        border: 4px solid rgb(110, 189, 242, 1);
      }
    }
  }

  .title {
    text-align: center;
    font-size: 1.5em;
    margin-bottom: 10px;
  }

  .answer-text {
    font-size: 1.3em;
    text-wrap: auto;
    font-family: "Scheherazade New";
  }

  .loader {
    margin: 40px auto;
  }

  .answer-references {
    font-size: 1.3em;
    list-style-type: none;
    padding-left: 0;

    .answer-reference {
      display: flex;
      flex-direction: row;
      margin-bottom: 20px;
    }

    .reference-header {
      font-weight: bold;
      text-transform: capitalize;
      padding: 15px;
      width: 100px;
      min-width: 100px;
      max-width: 100px;
    }

    .reference-content {
      display: flex;
      flex-direction: column;
      padding: 15px;
      flex-grow: 1;
    }

    .reference-persian {
      direction: rtl;
      text-align: right;
      padding-bottom: 5px;
    }

    .reference-header.masnavi {
      background-color: rgba(80, 215, 150, 1);
    }

    .reference-header.program {
      background-color: #eee;
    }

    .answer-reference.masnavi {
      border: 2px solid rgba(80, 215, 150, 1);
    }

    .answer-reference.program {
      border: 2px solid #eee;
    }

    .reference-header.ghazal {
      background-color: #F8C8DC;
    }

    .answer-reference.ghazal {
      border: 2px solid #F8C8DC;
    }

    span {
      display: block;
    }
  }

  .settings-container {
    max-width: 800px;
    font-family: system-ui, sans-serif;
    margin-top: 20px;
  }

  .collapsible-header {
    background-color: #f3f4f6;
    border: 1px solid #ccc;
    padding: 10px 15px;
    width: 100%;
    text-align: left;
    font-weight: bold;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .collapsible-content {
    border: 1px solid #ccc;
    border-top: none;
    padding: 10px 15px;
    background: #fff;
  }

  .settings-group {
    display: flex;
    align-items: center;

    h4 {
      width: 150px;
    }    
  }

  .checkbox-group {
    display: flex;
    flex-direction: row;
    gap: 0.5rem;
  }

  .fade-enter-active,
  .fade-leave-active {
    transition: all 0.2s ease;
  }
  .fade-enter-from,
  .fade-leave-to {
    opacity: 0;
    transform: translateY(-5px);
  }

  .button-select {
    display: flex;
    gap: 10px;
  }

  .button-select button {
    background: #f5f5f5;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 8px 16px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s ease;
  }

  .button-select button:hover {
    background: #eaeaea;
  }

  .button-select button.active {
    background-color: #007bff; /* Highlight color */
    color: white;
    border-color: #007bff;
  }  
}
</style>
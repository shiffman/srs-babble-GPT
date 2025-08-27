let generator = null;

window.addEventListener('DOMContentLoaded', init);

async function init() {
  // DOM elements
  const loader = document.getElementById('loader');
  const progressBar = document.getElementById('progress');
  const status = document.getElementById('status');
  const chatInput = document.getElementById('chat-input');
  const sendButton = document.getElementById('send-button');
  const messagesContainer = document.getElementById('messages');

  // Disable the send button until the model is loaded
  sendButton.disabled = true;

  try {
    // Automatically detect the available device
    const device = navigator.gpu ? 'webgpu' : 'wasm';
    // import Transformers.js library from CDN
    const { pipeline, env, TextStreamer } = await import(
      'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0'
    );
    const modelId = 'shiffman/model-gpt2-srs';

    // Show loading status to user
    setStatus(status, progressBar, 'initializing…', null);

    // Create the text generation pipeline with progress tracking
    // This loads the ONNX model, tokenizer, and sets up inference
    generator = await pipeline('text-generation', modelId, {
      device, // Use detected device (webgpu or wasm)
      dtype: 'fp32', // Use 32-bit floats
      progress_callback: (p) => {
        // Function called during loading
        const pct = 100 * p.progress;
        const label = p.status || 'loading…';
        setStatus(status, progressBar, label, pct);
      },
    });

    // Model loaded successfully!
    setStatus(status, progressBar, 'ready', 100);
    // Hide the loading UI
    loader.style.display = 'none';
    // Enable the send button
    sendButton.disabled = false;

    // Function to send a message
    async function sendMessage() {
      const text = chatInput.value.trim();
      if (!text || !generator) return;

      // Add user message to chat
      const userMessage = document.createElement('div');
      userMessage.className = 'message user-message';
      userMessage.textContent = text;
      messagesContainer.appendChild(userMessage);

      // Clear input and disable send button
      chatInput.value = '';
      sendButton.disabled = true;

      // Create bot message container
      const botMessage = document.createElement('div');
      botMessage.className = 'message bot-message';
      messagesContainer.appendChild(botMessage);

      // Scroll to bottom
      messagesContainer.scrollTop = messagesContainer.scrollHeight;

      try {
        // Configure text generation parameters with streaming
        const opts = {
          max_new_tokens: 80,
          do_sample: true,
          temperature: 0.7,
          top_p: 0.9,
          top_k: 0,
          repetition_penalty: 1.1,
          no_repeat_ngram_size: 4,
          streamer: new TextStreamer(generator.tokenizer, {
            skip_prompt: true,
            callback_function: (token) => {
              botMessage.textContent += token;
              messagesContainer.scrollTop = messagesContainer.scrollHeight;
            },
          }),
        };

        // Generate text using the model with streaming
        await generator(text, opts);
      } catch (err) {
        console.error(err);
        botMessage.textContent = 'Oops! Baby bot had a hiccup 👶';
      } finally {
        sendButton.disabled = false;
      }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !sendButton.disabled) {
        sendMessage();
      }
    });
  } catch (err) {
    console.error(err);
    setStatus(status, progressBar, 'error — see console', null);
  }
}

// Update the loading status and progress bar
function setStatus(el, bar, label, percentOrNull) {
  el.textContent = label;
  bar.style.width = `${Math.max(0, Math.min(100, percentOrNull))}%`;
  bar.style.animation = 'none';
}

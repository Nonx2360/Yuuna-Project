const display = document.getElementById('message-display');
const input = document.getElementById('user-input');
const btn = document.getElementById('send-btn');
const hero = document.getElementById('hero-section');
const ttsToggle = document.getElementById('tts-toggle');
const moodSelector = document.getElementById('mood-selector');
const modelSelector = document.querySelector('.model-selector'); // Select class instead of ID

// Modal Elements
const charModal = document.getElementById('character-modal');
const closeModal = document.querySelector('.close-modal');
const charListContainer = document.querySelector('.character-list-container');
const createCharForm = document.getElementById('create-char-form');
const charList = document.getElementById('character-list');
const createCharBtn = document.getElementById('create-char-btn');
const cancelCreateBtn = document.getElementById('cancel-create-btn');
const saveCharBtn = document.getElementById('save-char-btn');
const generatePromptBtn = document.getElementById('generate-prompt-btn');

// Form Inputs
const charNameInput = document.getElementById('char-name');
const charDescInput = document.getElementById('char-desc');
const charGenPromptInput = document.getElementById('char-gen-prompt');
const charPromptInput = document.getElementById('char-prompt');

let messageHistory = [];
let currentAudio = null;
let currentCharacter = null;
let characters = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadCharacters();
});

// Auto-resize textarea
input.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

async function speakText(text) {
    if (!text) return;

    // Stop any current audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }

    const speakerId = moodSelector.value;

    try {
        const response = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, speaker: speakerId })
        });

        if (response.status === 503) {
            alert("VOICEVOX engine is not running. Please start it on port 50021 to hear the voice!");
            return;
        }
        if (!response.ok) throw new Error('TTS failed');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        currentAudio = new Audio(url);
        currentAudio.play();
    } catch (error) {
        console.error('TTS Playback error:', error);
    }
}

function appendMessage(role, content, duration = null) {
    if (hero) display.classList.add('has-messages');

    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';

    if (role === 'user') {
        avatar.innerHTML = `<img src="/static/img/userProfile.jpg" alt="U">`;
    } else if (role === 'assistant') {
        // Use current character avatar if available
        const avatarSrc = currentCharacter ? currentCharacter.avatar : "/static/img/gptProfile.png";
        avatar.innerHTML = `<img src="${avatarSrc}" alt="AI" onerror="this.src='/static/img/gptProfile.png'">`;
    } else {
        avatar.innerText = 'S';
    }

    const textDiv = document.createElement('div');
    textDiv.className = 'text';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (role === 'user' || role === 'system') {
        bubble.textContent = content;
    } else {
        bubble.innerHTML = marked.parse(content);
    }
    textDiv.appendChild(bubble);

    // AI Actions Menu
    if (role === 'assistant') {
        const actions = document.createElement('div');
        actions.className = 'message-actions';

        const copyBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>', 'Copy');
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(bubble.innerText);
            copyBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
            setTimeout(() => copyBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>', 2000);
        };

        const voiceBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>', 'Read aloud');
        voiceBtn.onclick = () => speakText(bubble.innerText);

        const upBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>', 'Good response');
        const downBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>', 'Bad response');
        const shareBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg>', 'Share');
        const regenBtn = createActionButton('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>', 'Regenerate');

        actions.appendChild(copyBtn);
        actions.appendChild(upBtn);
        actions.appendChild(downBtn);
        actions.appendChild(shareBtn);
        actions.appendChild(regenBtn);
        actions.appendChild(voiceBtn);
        textDiv.appendChild(actions);
    }

    if (duration !== null) {
        const timeDiv = document.createElement('div');
        timeDiv.className = 'response-time';
        timeDiv.textContent = `Response time: ${duration}s`;
        textDiv.appendChild(timeDiv);
    }

    row.appendChild(avatar);
    row.appendChild(textDiv);
    display.appendChild(row);
    display.scrollTop = display.scrollHeight;

    return { row, bubble, textDiv };
}

function createActionButton(svg, title) {
    const btn = document.createElement('button');
    btn.className = 'action-icon-btn';
    btn.innerHTML = svg;
    btn.title = title;
    return btn;
}

async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;

    appendMessage('user', text);
    input.value = '';
    input.style.height = 'auto'; // Reset height
    messageHistory.push({ role: 'user', content: text });

    const { bubble: assistantBubble, textDiv: assistantTextDiv } = appendMessage('assistant', '');
    assistantBubble.textContent = '...';
    display.scrollTop = display.scrollHeight;

    try {
        const payload = {
            messages: messageHistory,
            system_prompt: currentCharacter ? currentCharacter.system_prompt : undefined,
            character_id: currentCharacter ? currentCharacter.id : 'default'
        };

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let duration = null;

        assistantBubble.textContent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });

            if (chunk.includes('__DURATION__')) {
                const parts = chunk.split('__DURATION__');
                fullResponse += parts[0];
                duration = parts[1];

                assistantBubble.innerHTML = marked.parse(fullResponse.trim());

                const timeDiv = document.createElement('div');
                timeDiv.className = 'response-time';
                timeDiv.textContent = `Response time: ${duration}s`;
                assistantTextDiv.appendChild(timeDiv);
            } else {
                fullResponse += chunk;
                assistantBubble.innerHTML = marked.parse(fullResponse);
            }

            display.scrollTop = display.scrollHeight;
        }

        const finalContent = fullResponse.trim();
        messageHistory.push({ role: 'assistant', content: finalContent });

        // Auto-speak if toggled on
        if (ttsToggle.checked) {
            speakText(finalContent);
        }

    } catch (error) {
        console.error('Fetch error:', error);
        assistantBubble.textContent = 'Error: Could not connect to the server.';
    }
}

btn.addEventListener('click', sendMessage);
input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Sidebar Toggle for Mobile
const menuToggle = document.getElementById('menu-toggle');
const sidebar = document.querySelector('.sidebar');
const sidebarOverlay = document.getElementById('sidebar-overlay');

function toggleSidebar() {
    sidebar.classList.toggle('open');
    if (sidebarOverlay) sidebarOverlay.classList.toggle('active');
}

if (menuToggle) {
    menuToggle.addEventListener('click', toggleSidebar);
}

if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', toggleSidebar);
}

// Close sidebar when clicking 'New Chat' on mobile
const newChatBtn = document.querySelector('.new-chat-btn');
if (newChatBtn) {
    newChatBtn.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('open');
            if (sidebarOverlay) sidebarOverlay.classList.remove('active');
        }
    });
}

// ============================================
// Character Management Logic
// ============================================

async function loadCharacters() {
    try {
        const response = await fetch('/api/characters');
        characters = await response.json();
        renderCharacterList();

        // Select default Character if none selected
        if (!currentCharacter && characters.length > 0) {
            selectCharacter(characters[0]);
        }
    } catch (error) {
        console.error('Failed to load characters:', error);
    }
}

function renderCharacterList() {
    charList.innerHTML = '';
    characters.forEach(char => {
        const card = document.createElement('div');
        card.className = `character-card ${currentCharacter && currentCharacter.id === char.id ? 'selected' : ''}`;

        // Main Click to Select
        card.onclick = (e) => {
            // Prevent selection if delete button was clicked
            if (e.target.closest('.delete-char-btn')) return;
            selectCharacter(char);
        };

        let deleteBtnHtml = '';
        if (char.id !== 'default') {
            deleteBtnHtml = `
                <button class="delete-char-btn" title="Delete Character" onclick="deleteCharacter(event, '${char.id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
            `;
        }

        card.innerHTML = `
            <img src="${char.avatar}" alt="${char.name}" class="char-avatar" onerror="this.src='/static/img/gptProfile.png'">
            <div class="char-info">
                <h3>${char.name}</h3>
                <p>${char.description || 'No description'}</p>
            </div>
            ${deleteBtnHtml}
        `;
        charList.appendChild(card);
    });
}

window.deleteCharacter = async function (event, charId) {
    event.stopPropagation(); // Stop card selection

    if (!confirm('Are you sure you want to delete this character?')) return;

    try {
        const response = await fetch(`/api/characters/${charId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            // If deleting current character, switch to default
            if (currentCharacter && currentCharacter.id === charId) {
                const defaultChar = characters.find(c => c.id === 'default');
                selectCharacter(defaultChar);
            }
            loadCharacters(); // Refresh list
        } else {
            alert('Failed to delete character');
        }
    } catch (error) {
        console.error('Error deleting character:', error);
        alert('Error deleting character');
    }
};

function selectCharacter(char) {
    currentCharacter = char;

    // Update UI
    modelSelector.innerHTML = `${char.name} <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"></path></svg>`;

    // Update list selection highlight
    Array.from(charList.children).forEach(child => child.classList.remove('selected'));
    // Re-render to update selection class properly or just update DOM
    renderCharacterList();

    // Close modal
    charModal.classList.remove('active');

    // Clear chat if switching characters (optional, but good practice)
    // display.innerHTML = ''; 
    // messageHistory = [];
    // if (hero) display.classList.add('has-messages'); // Or remove hero-section handling logic reset
}

// Modal handling
modelSelector.addEventListener('click', () => {
    charModal.classList.add('active');
    loadCharacters(); // Refresh list
});

closeModal.addEventListener('click', () => {
    charModal.classList.remove('active');
    resetCreateForm();
});

window.addEventListener('click', (e) => {
    if (e.target === charModal) {
        charModal.classList.remove('active');
        resetCreateForm();
    }
});

// Create Character Logic
createCharBtn.addEventListener('click', () => {
    charListContainer.style.display = 'none';
    createCharForm.style.display = 'block';
});

cancelCreateBtn.addEventListener('click', resetCreateForm);

function resetCreateForm() {
    createCharForm.style.display = 'none';
    charListContainer.style.display = 'block';
    charNameInput.value = '';
    charDescInput.value = '';
    charGenPromptInput.value = `Create a detailed system prompt for a character who is...
[Describe personality, traits, and background here]

The character should speak in a...
[Describe speaking style here]

Output only the system prompt.`;
    charPromptInput.value = '';
}

generatePromptBtn.addEventListener('click', async () => {
    const instruction = charGenPromptInput.value.trim();
    if (!instruction) {
        alert('Please enter generation instructions.');
        return;
    }

    generatePromptBtn.disabled = true;
    generatePromptBtn.innerHTML = 'Generating...';

    try {
        const response = await fetch('/api/generate_prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ instruction })
        });

        const data = await response.json();
        if (data.system_prompt) {
            charPromptInput.value = data.system_prompt;
        } else {
            alert('Failed to generate prompt: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error generating prompt:', error);
        alert('Error generating prompt');
    } finally {
        generatePromptBtn.disabled = false;
        generatePromptBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path></svg> Generate with Yuuna';
    }
});

saveCharBtn.addEventListener('click', async () => {
    const name = charNameInput.value.trim();
    const system_prompt = charPromptInput.value.trim();
    const description = charDescInput.value.trim();

    if (!name || !system_prompt) {
        alert('Name and System Prompt are required.');
        return;
    }

    saveCharBtn.disabled = true;

    try {
        const response = await fetch('/api/characters', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                system_prompt,
                description,
                avatar: 'static/img/gptProfile.png' // Default avatar for now
            })
        });

        const newChar = await response.json();
        characters.push(newChar);
        selectCharacter(newChar);
        resetCreateForm();

    } catch (error) {
        console.error('Error saving character:', error);
        alert('Failed to save character');
    } finally {
        saveCharBtn.disabled = false;
    }
});

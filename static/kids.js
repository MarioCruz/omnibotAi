        let isRunning = false;
        const socket = io();
        const brainEl = document.getElementById('robotBrain');
        let brainStarted = false;

        function escHtml(s) {
            return String(s == null ? '' : s)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        socket.on('update', (data) => {
            // Show what robot sees
            if (data.detections && data.detections.length > 0) {
                const det = data.detections[0];
                const label = det.label;
                const conf = (det.confidence * 100) | 0;
                const icon = label === 'person' ? '👤' : label === 'cat' ? '🐱' : label === 'dog' ? '🐶' : '📦';

                // Show nav decision if available
                const nav = data.llm_debug;
                if (nav && nav.target) {
                    if (!brainStarted) { brainEl.innerHTML = ''; brainStarted = true; }
                    const cmds = nav.parsed_commands || [];
                    let actionText = 'LOOK';
                    let actionClass = '';
                    if (cmds.length > 0) {
                        const cmd = cmds[0];
                        if (cmd.includes('forward')) { actionText = 'GO!'; actionClass = 'forward'; }
                        else if (cmd.includes('left')) { actionText = 'LEFT'; actionClass = 'left'; }
                        else if (cmd.includes('right')) { actionText = 'RIGHT'; actionClass = 'right'; }
                        else if (cmd.includes('stop')) { actionText = 'STOP'; actionClass = 'stop'; }
                    }
                    const entry = document.createElement('div');
                    entry.className = 'brain-entry';
                    entry.innerHTML =
                        '<span class="brain-icon">' + icon + '</span>' +
                        '<span class="brain-target">' + escHtml(label) + ' ' + conf + '%</span>' +
                        '<span class="brain-action ' + escHtml(actionClass) + '">' + escHtml(actionText) + '</span>' +
                        '<span class="brain-detail">' + escHtml((nav.response || '').substring(0, 40)) + '</span>';
                    brainEl.prepend(entry);
                    while (brainEl.children.length > 20) brainEl.lastChild.remove();
                }
            }
        });

        function togglePower() {
            const btn = document.getElementById('powerBtn');
            const powerLed = document.getElementById('powerLed');
            const statusLed = document.getElementById('statusLed');
            const statusText = document.getElementById('statusText');
            const modeText = document.getElementById('modeText');

            if (!isRunning) {
                fetch('/api/start', { method: 'POST' });
                btn.textContent = '[ DEACTIVATE ROBOT ]';
                btn.className = 'power-btn stop';
                powerLed.className = 'led power on';
                statusLed.className = 'led status on';
                statusText.textContent = 'ONLINE';
                statusText.className = 'status-text';
                modeText.textContent = 'ACTIVE';
                modeText.className = 'status-text';
                isRunning = true;
            } else {
                fetch('/api/stop', { method: 'POST' });
                btn.textContent = '[ ACTIVATE ROBOT ]';
                btn.className = 'power-btn start';
                powerLed.className = 'led power';
                statusLed.className = 'led status';
                statusText.textContent = 'STANDBY';
                statusText.className = 'status-text off';
                modeText.textContent = 'IDLE';
                modeText.className = 'status-text off';
                isRunning = false;
            }
        }

        function startMission(displayName, taskText) {
            const missionEl = document.getElementById('missionText');
            missionEl.textContent = displayName;
            missionEl.className = 'mission-text';
            fetch('/api/task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task: taskText })
            });

            // Auto-start if not running
            if (!isRunning) {
                togglePower();
            }
        }

        // Shared helper so "busy" feedback is consistent across kid buttons.
        // Without this, a 409 silently drops and it looks like the click did nothing.
        function kidCommand(cmd, missionLabel) {
            const missionEl = document.getElementById('missionText');
            if (missionLabel) {
                missionEl.textContent = missionLabel;
                missionEl.className = 'mission-text';
            }
            return fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: cmd })
            })
                .then(r => r.json().then(data => ({ status: r.status, data })))
                .then(({ status, data }) => {
                    if (status === 409 || data.status === 'busy') {
                        missionEl.textContent = 'BUSY — TRY AGAIN';
                        missionEl.className = 'mission-text';
                    }
                })
                .catch(() => {
                    missionEl.textContent = 'ERROR';
                    missionEl.className = 'mission-text';
                });
        }

        function drive(direction) {
            kidCommand(direction, null);
        }

        function doDance() {
            kidCommand('dance', 'DANCE MODE');
        }

        function sayHello() {
            kidCommand('phrase("omnibot")', 'VOICE OUTPUT');
        }

        function sayPhrase(name) {
            kidCommand('phrase("' + name + '")', 'SAYING: ' + name.toUpperCase());
        }

        function speakerOff() {
            kidCommand('speaker_off', 'SPEAKER OFF');
        }

        function endMission() {
            const missionEl = document.getElementById('missionText');
            missionEl.textContent = 'AWAITING ORDERS...';
            missionEl.className = 'mission-text none';
            fetch('/api/task/end', { method: 'POST' });
        }

        function describeScene() {
            const missionEl = document.getElementById('missionText');
            const speakLocal = document.getElementById('speakLocal').checked;
            missionEl.textContent = 'THINKING...';
            missionEl.className = 'mission-text';
            fetch('/api/describe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speak_robot: !speakLocal })
            })
                .then(r => r.json())
                .then(data => {
                    const text = data.description || data.error || 'NO RESPONSE';
                    missionEl.textContent = text;
                    if (speakLocal && window.speechSynthesis) {
                        window.speechSynthesis.cancel();
                        const utter = new SpeechSynthesisUtterance(text);
                        utter.rate = 1.0;
                        utter.pitch = 0.9;
                        window.speechSynthesis.speak(utter);
                    }
                    setTimeout(() => {
                        missionEl.textContent = 'AWAITING ORDERS...';
                        missionEl.className = 'mission-text none';
                    }, 10000);
                })
                .catch(() => { missionEl.textContent = 'ERROR'; });
        }

        function checkBluetooth() {
            fetch('/api/bluetooth')
                .then(r => r.json())
                .then(data => {
                    const led = document.getElementById('btLed');
                    const text = document.getElementById('btText');
                    if (data.connected) {
                        led.className = 'led bluetooth on';
                        text.textContent = data.devices[0] || 'BT ON';
                        text.className = 'status-text';
                    } else {
                        led.className = 'led bluetooth';
                        text.textContent = 'BT OFF';
                        text.className = 'status-text off';
                    }
                })
                .catch(() => {});
        }

        checkBluetooth();
        setInterval(checkBluetooth, 15000);

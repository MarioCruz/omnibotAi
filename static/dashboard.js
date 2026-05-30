        const socket = io();
        const MAX_HISTORY = 20;
        const MAX_NAV_LOG = 30;
        const detectionsList = document.getElementById('detectionsList');

        // Escape dynamic data before inserting via innerHTML. Labels come from
        // the detector and are normally safe, but a custom model or bad actor
        // could inject markup — cheap to be defensive.
        function escHtml(s) {
            return String(s == null ? '' : s)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }
        const navLogEl = document.getElementById('navLog');
        const activityLog = document.getElementById('activityLog');
        let navLogStarted = false;

        socket.on('connect', () => {
            log('Connected to server', 'success');
            updateStatus('Connected', false);
        });

        socket.on('disconnect', () => {
            log('Disconnected from server', 'error');
            updateStatus('Disconnected', false);
        });

        socket.on('update', (data) => {
            // Update stat numbers directly (no innerHTML rebuild)
            document.getElementById('statIterations').textContent = data.stats.iterations;
            document.getElementById('statFps').textContent = data.stats.fps;
            document.getElementById('statDetections').textContent = data.stats.total_detections;
            document.getElementById('statCommands').textContent = data.stats.total_commands;

            // Detection history — only add new entry if there are detections
            if (data.detections.length > 0) {
                const time = new Date().toLocaleTimeString();
                const summary = data.detections.map(d =>
                    escHtml(d.label) + ' (' + (d.confidence * 100 | 0) + '%)'
                ).join(', ');
                const entry = document.createElement('div');
                entry.className = 'log-entry info';
                entry.innerHTML = '<span style="color:#888">[' + escHtml(time) + ']</span> <span style="color:#00ff88">' + data.detections.length + 'x</span> ' + summary;
                detectionsList.prepend(entry);
                while (detectionsList.children.length > MAX_HISTORY) detectionsList.lastChild.remove();
            }

            // Navigation log — show real-time decisions
            const nav = data.llm_debug;
            if (nav && nav.target) {
                if (!navLogStarted) {
                    navLogEl.innerHTML = '';
                    navLogStarted = true;
                }
                const time = nav.timestamp || new Date().toLocaleTimeString();
                // Figure out action type for color coding
                const cmds = nav.parsed_commands || [];
                let actionText = 'idle';
                let actionClass = '';
                if (cmds.length > 0) {
                    const cmd = cmds[0];
                    if (cmd.includes('forward')) { actionText = 'FORWARD'; actionClass = 'forward'; }
                    else if (cmd.includes('left')) { actionText = 'LEFT'; actionClass = 'left'; }
                    else if (cmd.includes('right')) { actionText = 'RIGHT'; actionClass = 'right'; }
                    else if (cmd.includes('stop')) { actionText = 'STOP'; actionClass = 'stop'; }
                }
                const entry = document.createElement('div');
                entry.className = 'nav-entry';
                entry.innerHTML =
                    '<span class="nav-time">' + escHtml(time) + '</span>' +
                    '<span class="nav-target">' + escHtml(nav.target || '') + '</span>' +
                    '<span class="nav-action ' + escHtml(actionClass) + '">' + escHtml(actionText) + '</span>' +
                    '<span class="nav-reason">' + escHtml(nav.response || '') + '</span>';
                navLogEl.prepend(entry);
                while (navLogEl.children.length > MAX_NAV_LOG) navLogEl.lastChild.remove();
            }
        });

        function updateStatus(text, running) {
            const el = document.getElementById('systemStatus');
            el.textContent = text;
            el.className = 'status ' + (running ? 'running' : 'stopped');
        }

        function log(msg, type) {
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + (type || '');
            entry.textContent = '[' + time + '] ' + msg;
            activityLog.prepend(entry);
            while (activityLog.children.length > 30) activityLog.lastChild.remove();
        }

        function startSystem() {
            fetch('/api/start', { method: 'POST' }).then(() => {
                log('System started', 'success');
                updateStatus('Running', true);
            });
        }

        function stopSystem() {
            fetch('/api/stop', { method: 'POST' }).then(() => {
                log('System stopped', 'error');
                updateStatus('Stopped', false);
                navLogStarted = false;
            });
        }

        function pauseSystem() {
            fetch('/api/pause', { method: 'POST' }).then(() => {
                log('System paused', 'info');
                updateStatus('Paused', false);
            });
        }

        function setTask() {
            const task = document.getElementById('taskInput').value;
            fetch('/api/task', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task: task })
            }).then(() => log('Task set: ' + task, 'info'));
        }

        function endTask() {
            fetch('/api/task/end', { method: 'POST' }).then(() => {
                log('Task ended', 'info');
            });
        }

        function describeScene() {
            const speakLocal = document.getElementById('speakLocal').checked;
            log('Asking robot to describe what it sees...', 'info');
            fetch('/api/describe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ speak_robot: !speakLocal })
            })
                .then(r => r.json())
                .then(data => {
                    if (data.description) {
                        log('Robot says: ' + data.description, 'success');
                        if (speakLocal && window.speechSynthesis) {
                            window.speechSynthesis.cancel();
                            const utter = new SpeechSynthesisUtterance(data.description);
                            utter.rate = 1.0;
                            utter.pitch = 0.9;
                            window.speechSynthesis.speak(utter);
                        }
                    } else if (data.error) {
                        log('Describe error: ' + data.error, 'error');
                    }
                })
                .catch(() => log('Describe request failed', 'error'));
        }

        function sendCommand(cmd) {
            fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: cmd })
            })
                .then(r => r.json().then(data => ({ status: r.status, data })))
                .then(({ status, data }) => {
                    if (status === 409 || data.status === 'busy') {
                        log('Busy, try again: ' + cmd, 'error');
                    } else if (data.status === 'error') {
                        log('Error (' + cmd + '): ' + (data.message || 'unknown'), 'error');
                    } else {
                        log('Command: ' + cmd, 'info');
                    }
                })
                .catch(() => log('Command failed: ' + cmd, 'error'));
        }

        function speakText() {
            const text = document.getElementById('speechInput').value;
            if (text) sendCommand('speakText("' + text.replace(/"/g, '') + '")');
        }

        function playPhrase(name) {
            sendCommand('phrase("' + name + '")');
        }

        // Bluetooth check — every 15s is plenty
        function checkBt() {
            fetch('/api/bluetooth').then(r => r.json()).then(data => {
                const el = document.getElementById('statBt');
                el.textContent = data.connected ? (data.devices[0] || 'ON') : 'OFF';
                el.style.color = data.connected ? '#4488ff' : '#ff4444';
            }).catch(() => {});
        }
        checkBt();
        setInterval(checkBt, 15000);

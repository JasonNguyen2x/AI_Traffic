// Traffic AI Frontend Application

class TrafficAIApp {
    constructor() {
        this.ws = null;
        this.wsConnected = false;
        this.streamActive = false;

        // Density thresholds (default from config)
        this.densityThresholds = {
            low: 5,
            medium: 10
        };

        // Load saved thresholds from localStorage
        this.loadDensitySettings();

        // DOM elements
        this.connectBtn = document.getElementById('connectBtn');
        this.disconnectBtn = document.getElementById('disconnectBtn');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.errorMessage = document.getElementById('errorMessage');
        this.videoFrame = document.getElementById('videoFrame');
        this.noVideoMessage = document.getElementById('noVideoMessage');
        this.frameCount = document.getElementById('frameCount');

        // Settings modal elements
        this.settingsModal = document.getElementById('settingsModal');
        this.closeSettings = document.getElementById('closeSettings');
        this.saveSettings = document.getElementById('saveSettings');
        this.resetSettings = document.getElementById('resetSettings');
        this.densityLowInput = document.getElementById('densityLow');
        this.densityMediumInput = document.getElementById('densityMedium');

        // ponytail: statElements removed - HTML has no stats panel, only density & plates sidebar

        this.platesList = document.getElementById('platesList');

        // Ensure initial state
        this.showNoVideo();

        // Bind events
        this.connectBtn.addEventListener('click', () => this.connect());
        this.disconnectBtn.addEventListener('click', () => this.disconnect());

        // Settings modal events
        this.closeSettings.addEventListener('click', () => this.closeSettingsModal());
        this.saveSettings.addEventListener('click', () => this.saveDensitySettings());
        this.resetSettings.addEventListener('click', () => this.resetDensitySettings());

        // Keyboard shortcut - Press 'G' to open settings
        document.addEventListener('keydown', (e) => {
            if (e.key === 'g' || e.key === 'G') {
                // Don't trigger if user is typing in an input
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                    return;
                }
                e.preventDefault();
                this.openSettings();
            }
            // ESC to close modal
            if (e.key === 'Escape' && this.settingsModal.style.display === 'flex') {
                this.closeSettingsModal();
            }
        });

        // Update preview when inputs change
        this.densityLowInput.addEventListener('input', () => this.updatePreview());
        this.densityMediumInput.addEventListener('input', () => this.updatePreview());

        // Close modal when clicking outside
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) {
                this.closeSettingsModal();
            }
        });

        // Protocol change handler
        document.getElementById('protocol').addEventListener('change', (e) => {
            this.handleProtocolChange(e.target.value);
        });

        // Initialize WebSocket
        this.initWebSocket();

        // Poll statistics periodically
        setInterval(() => this.updateStatistics(), 1000);
    }

    loadDensitySettings() {
        const saved = localStorage.getItem('densityThresholds');
        if (saved) {
            try {
                this.densityThresholds = JSON.parse(saved);
            } catch (e) {
                console.error('Failed to load density settings:', e);
            }
        }
    }

    saveDensitySettingsToStorage() {
        localStorage.setItem('densityThresholds', JSON.stringify(this.densityThresholds));
    }

    openSettings() {
        // Load current values
        this.densityLowInput.value = this.densityThresholds.low;
        this.densityMediumInput.value = this.densityThresholds.medium;
        this.updatePreview();

        // Show modal with default tab
        this.settingsModal.style.display = 'flex';
        
        // Use setTimeout to ensure DOM is ready
        setTimeout(() => {
            // Setup tab listeners if not already done
            const tabBtns = document.querySelectorAll('.tab-btn');
            tabBtns.forEach(btn => {
                // Remove old listeners by cloning
                const newBtn = btn.cloneNode(true);
                btn.parentNode.replaceChild(newBtn, btn);
                
                // Add new listener
                newBtn.addEventListener('click', (e) => {
                    const tabName = e.currentTarget.getAttribute('data-tab');
                    console.log('Tab clicked:', tabName);
                    this.switchTab(tabName);
                });
            });
            
            // Activate default tab
            this.switchTab('connection');
        }, 0);
    }

    switchTab(tabName) {
        console.log('Switching to tab:', tabName);
        
        // Remove active class from all tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
            content.style.display = 'none';
        });

        // Add active class to selected tab
        const tabBtn = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
        const tabContent = document.getElementById(`${tabName}Tab`);
        
        if (tabBtn) {
            tabBtn.classList.add('active');
            console.log('Activated tab button:', tabName);
        } else {
            console.error('Tab button not found:', tabName);
        }
        
        if (tabContent) {
            tabContent.classList.add('active');
            tabContent.style.display = 'block';
            console.log('Activated tab content:', tabName);
        } else {
            console.error('Tab content not found:', tabName + 'Tab');
        }
    }

    closeSettingsModal() {
        this.settingsModal.style.display = 'none';
    }

    updatePreview() {
        const low = parseInt(this.densityLowInput.value) || 5;
        const medium = parseInt(this.densityMediumInput.value) || 10;

        document.getElementById('previewLow').textContent = low;
        document.getElementById('previewLow2').textContent = low;
        document.getElementById('previewMedium').textContent = medium;
        document.getElementById('previewMedium2').textContent = medium;
    }

    saveDensitySettings() {
        const low = parseInt(this.densityLowInput.value);
        const medium = parseInt(this.densityMediumInput.value);

        // Validation
        if (low < 1 || medium < 1) {
            alert('Thresholds must be at least 1');
            return;
        }

        if (low >= medium) {
            alert('Low threshold must be less than medium threshold');
            return;
        }

        // Save
        this.densityThresholds.low = low;
        this.densityThresholds.medium = medium;
        this.saveDensitySettingsToStorage();

        // Close modal
        this.closeSettingsModal();

        // Show success message
        this.showSuccess('✅ Density thresholds updated successfully!');
    }

    resetDensitySettings() {
        // Reset to defaults
        this.densityThresholds = { low: 5, medium: 10 };
        this.densityLowInput.value = 5;
        this.densityMediumInput.value = 10;
        this.updatePreview();
        this.saveDensitySettingsToStorage();

        this.showSuccess('🔄 Reset to default values');
    }

    showSuccess(message) {
        // Create temporary success message
        const successDiv = document.createElement('div');
        successDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #10b981;
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 2000;
            animation: slideInRight 0.3s;
        `;
        successDiv.textContent = message;
        document.body.appendChild(successDiv);

        setTimeout(() => {
            successDiv.style.animation = 'slideOutRight 0.3s';
            setTimeout(() => successDiv.remove(), 300);
        }, 2000);
    }

    showNoVideo() {
        this.videoFrame.style.display = 'none';
        this.noVideoMessage.style.display = 'flex';

        // Hide overlays
        const densityOverlay = document.getElementById('densityOverlay');
        if (densityOverlay) {
            densityOverlay.style.display = 'none';
        }
        
        const fpsOverlay = document.getElementById('fpsOverlay');
        if (fpsOverlay) {
            fpsOverlay.style.display = 'none';
        }
    }

    showVideo() {
        this.videoFrame.style.display = 'block';
        this.noVideoMessage.style.display = 'none';

        // Show overlays
        const densityOverlay = document.getElementById('densityOverlay');
        if (densityOverlay) {
            densityOverlay.style.display = 'block';
        }
    }

    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.wsConnected = true;
            };

            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.wsConnected = false;
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.wsConnected = false;

                // Reconnect after 3 seconds
                setTimeout(() => this.initWebSocket(), 3000);
            };

        } catch (error) {
            console.error('Failed to create WebSocket:', error);
        }
    }

    handleWebSocketMessage(event) {
        // Check if message is binary (frame) or text (JSON)
        if (event.data instanceof Blob) {
            // Binary frame - TODO: Phase 10
            this.displayFrame(event.data);
        } else {
            // JSON message
            try {
                const msg = JSON.parse(event.data);

                if (msg.type === 'stats') {
                    this.updateStatsFromMessage(msg);
                } else if (msg.type === 'plate_detection') {
                    console.log('Received plate_detection:', msg);
                    this.addPlateDetection(msg);
                } else if (msg.type === 'plate') {
                    // Legacy support
                    this.addPlateDetection(msg);
                }
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        }
    }

    displayFrame(blob) {
        // Convert blob to image URL and display
        const url = URL.createObjectURL(blob);

        // Revoke previous URL to prevent memory leak
        if (this.videoFrame.src && this.videoFrame.src.startsWith('blob:')) {
            URL.revokeObjectURL(this.videoFrame.src);
        }

        this.videoFrame.src = url;

        // Show video, hide "no video" message
        this.showVideo();
    }

    handleProtocolChange(protocol) {
        const urlGroup = document.getElementById('urlGroup');
        const fileGroup = document.getElementById('fileGroup');
        const streamUrlInput = document.getElementById('streamUrl');

        if (protocol === 'video') {
            // Hide URL field, show file field
            urlGroup.style.display = 'none';
            fileGroup.style.display = 'block';
        } else {
            // Show URL field, hide file field
            urlGroup.style.display = 'block';
            fileGroup.style.display = 'none';

            // Update placeholder based on protocol
            if (protocol === 'rtsp') {
                streamUrlInput.placeholder = 'rtsp://username:password@192.168.1.100:554/stream1';
            } else if (protocol === 'hls') {
                streamUrlInput.placeholder = 'http://192.168.1.100:8080/live/stream.m3u8';
            }
        }
    }

    async connect() {
        const protocol = document.getElementById('protocol').value;

        let url;

        if (protocol === 'video') {
            // Video file mode - upload first
            const fileInput = document.getElementById('videoFile');
            const file = fileInput.files[0];

            if (!file) {
                this.showError('Please select a video file');
                return;
            }

            // Update UI
            this.connectBtn.disabled = true;
            this.disconnectBtn.disabled = true;
            this.updateStatus('connecting', 'UPLOADING...');
            this.hideError();

            try {
                // Upload file first
                const formData = new FormData();
                formData.append('file', file);

                const uploadResponse = await fetch('/api/stream/upload', {
                    method: 'POST',
                    body: formData,
                });

                if (!uploadResponse.ok) {
                    const error = await uploadResponse.json();
                    throw new Error(error.detail || 'Upload failed');
                }

                const uploadResult = await uploadResponse.json();
                url = uploadResult.path;  // Use uploaded file path

                this.updateStatus('connecting', 'CONNECTING...');

            } catch (error) {
                console.error('Upload error:', error);
                this.showError('Upload failed: ' + error.message);
                this.updateStatus('error', 'ERROR');
                this.connectBtn.disabled = false;
                this.disconnectBtn.disabled = true;
                return;
            }
        } else {
            // Network stream mode - get URL directly
            url = document.getElementById('streamUrl').value.trim();

            if (!url) {
                this.showError('Stream URL is required');
                return;
            }

            // Basic URL validation
            if (protocol === 'rtsp' && !url.startsWith('rtsp://')) {
                this.showError('RTSP URL must start with rtsp://');
                return;
            }

            if (protocol === 'hls' && !url.startsWith('http://') && !url.startsWith('https://')) {
                this.showError('HLS URL must start with http:// or https://');
                return;
            }

            // Update UI
            this.connectBtn.disabled = true;
            this.disconnectBtn.disabled = true;
            this.updateStatus('connecting', 'CONNECTING...');
            this.hideError();
        }

        try {
            const response = await fetch('/api/stream/connect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    protocol,
                    url
                }),
            });

            const result = await response.json();

            if (result.success) {
                this.streamActive = true;
                this.updateStatus('connected', 'CONNECTED');
                this.connectBtn.disabled = true;
                this.disconnectBtn.disabled = false;

                console.log('Stream connected:', result.config);
            } else {
                this.showError(result.error || 'Connection failed');
                this.updateStatus('disconnected', 'DISCONNECTED');
                this.connectBtn.disabled = false;
                this.disconnectBtn.disabled = true;
            }

        } catch (error) {
            console.error('Connection error:', error);
            this.showError('Network error: ' + error.message);
            this.updateStatus('error', 'ERROR');
            this.connectBtn.disabled = false;
            this.disconnectBtn.disabled = true;
        }
    }

    async disconnect() {
        this.disconnectBtn.disabled = true;

        try {
            const response = await fetch('/api/stream/disconnect', {
                method: 'POST',
            });

            const result = await response.json();

            if (result.success) {
                this.streamActive = false;
                this.updateStatus('disconnected', 'DISCONNECTED');
                this.connectBtn.disabled = false;
                this.disconnectBtn.disabled = true;

                // Hide video, show "no video" message
                if (this.videoFrame.src && this.videoFrame.src.startsWith('blob:')) {
                    URL.revokeObjectURL(this.videoFrame.src);
                }
                this.showNoVideo();

                // Reset statistics
                this.resetStatistics();

                console.log('Stream disconnected');
            }

        } catch (error) {
            console.error('Disconnect error:', error);
            this.disconnectBtn.disabled = false;
        }
    }

    updateStatus(status, text) {
        this.statusIndicator.className = `status-dot status-${status}`;
        this.statusText.textContent = text;
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.style.display = 'block';
    }

    hideError() {
        this.errorMessage.style.display = 'none';
    }

    async updateStatistics() {
        // ponytail: Stats panel removed, polling disabled. WebSocket sends stats via 'stats' message type
        return;
    }

    updateStatsFromMessage(msg) {
        // ponytail: Stats panel removed from HTML, only update density & FPS overlays
        console.log('Stats received:', {
            fps: msg.fps,
            frame_time: msg.frame_time,
            current_vehicles: msg.current_vehicles,
            density_level: msg.density_level
        });

        // Update FPS display if available
        if (msg.fps !== undefined) {
            this.updateFPSDisplay(msg.fps, msg.frame_time);
        } else {
            console.warn('FPS data missing from stats');
        }

        // Update density indicator
        if (msg.current_vehicles !== undefined && msg.density_level) {
            this.updateDensity(msg.current_vehicles, msg.density_level);
        } else {
            console.warn('Density data missing:', {
                current_vehicles: msg.current_vehicles,
                density_level: msg.density_level
            });
        }
    }

    updateFPSDisplay(fps, frameTime) {
        console.log('updateFPSDisplay called:', fps, frameTime);
        
        // Update or create FPS overlay
        let fpsOverlay = document.getElementById('fpsOverlay');

        if (!fpsOverlay) {
            console.log('Creating FPS overlay');
            // Create FPS overlay
            fpsOverlay = document.createElement('div');
            fpsOverlay.id = 'fpsOverlay';
            fpsOverlay.style.cssText = `
                position: absolute;
                top: 20px;
                left: 20px;
                background-color: rgba(0, 0, 0, 0.7);
                color: #10b981;
                padding: 8px 12px;
                border-radius: 6px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                font-weight: 700;
                z-index: 10;
                backdrop-filter: blur(4px);
                border: 1px solid rgba(16, 185, 129, 0.3);
            `;

            const videoContainer = document.querySelector('.video-container');
            if (videoContainer) {
                videoContainer.appendChild(fpsOverlay);
                console.log('FPS overlay added to video container');
            } else {
                console.error('Video container not found!');
            }
        }

        // Color based on FPS
        let color = '#10b981'; // Green
        if (fps < 20) {
            color = '#ef4444'; // Red
        } else if (fps < 25) {
            color = '#f59e0b'; // Yellow
        }

        fpsOverlay.style.color = color;
        fpsOverlay.style.borderColor = color + '33';
        fpsOverlay.innerHTML = `FPS: ${fps.toFixed(1)} | ${frameTime}ms`;
        console.log('FPS overlay updated:', fpsOverlay.innerHTML);
    }

    updateDensity(count, level) {
        console.log('updateDensity called:', count, level);
        
        const badge = document.getElementById('densityBadge');
        const overlay = document.getElementById('densityOverlay');
        
        if (!badge) {
            console.error('densityBadge element not found!');
            return;
        }
        if (!overlay) {
            console.error('densityOverlay element not found!');
            return;
        }

        // Recalculate level based on current thresholds
        if (count < this.densityThresholds.low) {
            level = 'low';
        } else if (count <= this.densityThresholds.medium) {
            level = 'medium';
        } else {
            level = 'high';
        }

        console.log('Calculated level:', level, 'for count:', count);

        // Update badge style and text
        badge.className = 'density-badge density-' + level;

        // Format: "THÔNG THOÁNG | 5 xe"
        let statusText = '';
        if (level === 'low') {
            statusText = 'THÔNG THOÁNG';
        } else if (level === 'medium') {
            statusText = 'ĐÔNG';
        } else {
            statusText = 'ÙN TẮC';
        }
        
        badge.innerHTML = `${statusText} | ${count} xe`;
        
        // Ensure overlay is visible
        overlay.style.display = 'block';
        
        console.log('Density badge updated:', badge.innerHTML, 'display:', overlay.style.display);
    }

    addPlateDetection(msg) {
        // ponytail: New layout with vehicle + plate crops
        const plateItem = document.createElement('div');
        plateItem.className = 'plate-item';
        
        // Format: "Vehicle #N | plate_image | OCR text"
        plateItem.innerHTML = `
            <div class="plate-detection-row">
                <div class="vehicle-info">
                    <div class="vehicle-label">Vehicle #${msg.track_id}</div>
                    ${msg.vehicle_image ? `<img src="data:image/jpeg;base64,${msg.vehicle_image}" class="vehicle-thumb" alt="Vehicle">` : ''}
                </div>
                <div class="plate-info">
                    ${msg.plate_image ? `<img src="data:image/jpeg;base64,${msg.plate_image}" class="plate-thumb" alt="Plate">` : ''}
                    <div class="plate-text">${msg.text || 'N/A'}</div>
                </div>
            </div>
        `;

        // Remove "no data" message
        const noData = this.platesList.querySelector('.no-data');
        if (noData) {
            noData.remove();
        }

        // Insert at top
        this.platesList.insertBefore(plateItem, this.platesList.firstChild);

        // Limit to 30 items (images take more memory)
        while (this.platesList.children.length > 30) {
            this.platesList.removeChild(this.platesList.lastChild);
        }
    }

    updatePlatesList(plates) {
        if (plates.length === 0) return;

        // Clear existing items
        this.platesList.innerHTML = '';

        plates.forEach(plate => {
            const plateItem = document.createElement('div');
            plateItem.className = 'plate-item';
            plateItem.innerHTML = `
                <div class="plate-number">${plate.plate_number}</div>
                <div class="plate-meta">
                    <span>🚗 ID: ${plate.track_id}</span>
                    <span>📊 Conf: ${(plate.confidence * 100).toFixed(0)}%</span>
                    <span>🕐 ${plate.timestamp}</span>
                </div>
            `;
            this.platesList.appendChild(plateItem);
        });
    }

    resetStatistics() {
        Object.values(this.statElements).forEach(el => {
            el.textContent = '0';
        });

        this.frameCount.textContent = '0';

        this.platesList.innerHTML = '<div class="no-data">No plates detected yet</div>';
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Traffic AI App initializing...');
    const app = new TrafficAIApp();
    console.log('Traffic AI App ready');
});

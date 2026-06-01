document.addEventListener("DOMContentLoaded", () => {
    const uploadArea = document.getElementById("upload-area");
    const videoInput = document.getElementById("video-input");
    
    const uploadSection = document.getElementById("upload-section");
    const loadingSection = document.getElementById("loading-section");
    const resultsSection = document.getElementById("results-section");

    const resultVideo = document.getElementById("result-video");
    const verdictTitle = document.getElementById("verdict-title");
    const feedbackText = document.getElementById("feedback-text");
    
    const valReps = document.getElementById("val-reps");
    const valAvgq = document.getElementById("val-avgq");
    const valConsistency = document.getElementById("val-consistency");
    const valLowest = document.getElementById("val-lowest");
    
    const btnNew = document.getElementById("btn-new");
    let chartInstance = null;

    // --- Upload Handlers ---
    uploadArea.addEventListener("click", () => videoInput.click());

    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    videoInput.addEventListener("change", () => {
        if (videoInput.files.length) {
            handleFileUpload(videoInput.files[0]);
        }
    });

    btnNew.addEventListener("click", () => {
        resultsSection.classList.add("hidden");
        uploadSection.classList.remove("hidden");
        videoInput.value = ""; // reset
    });

    async function handleFileUpload(file) {
        if (!file.type.match('video.*')) {
            alert("Please upload a valid video file.");
            return;
        }

        // Show loading state
        uploadSection.classList.add("hidden");
        loadingSection.classList.remove("hidden");

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Note: Update URL based on where the fast API is hosted. Use absolute path if separated.
            // Using relative here expects the frontend to be served by the FastAPI app / mounted.
            const response = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const data = await response.json();
            displayResults(data);
            
        } catch (error) {
            console.error("Analysis Error:", error);
            alert("An error occurred during video analysis. Please try again.");
            loadingSection.classList.add("hidden");
            uploadSection.classList.remove("hidden");
        }
    }

    function displayResults(data) {
        loadingSection.classList.add("hidden");
        resultsSection.classList.remove("hidden");

        // Set video
        resultVideo.src = data.video_url;

        // Set feedback
        verdictTitle.innerText = data.verdict;
        feedbackText.innerText = data.feedback;
        
        if(data.verdict.includes("Excellent")){
            verdictTitle.style.color = "var(--success)";
        } else if(data.verdict.includes("Needs Improvement")) {
            verdictTitle.style.color = "var(--danger)";
        } else {
            verdictTitle.style.color = "var(--text-primary)";
        }

        // Set metrics
        valReps.innerText = data.rep_count;
        valAvgq.innerText = data.avg_q_score.toFixed(1);
        valConsistency.innerText = data.consistency.toFixed(1) + "%";
        valLowest.innerText = data.min_q_score.toFixed(1);

        // Render Chart
        renderChart(data.q_scores_over_time);
    }

    function renderChart(qScores) {
        const ctx = document.getElementById("qscore-chart").getContext("2d");
        
        if (chartInstance) {
            chartInstance.destroy();
        }

        const labels = Array.from({length: qScores.length}, (_, i) => i + 1);

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'value',
                    data: qScores,
                    borderColor: '#ff4b4b',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    fill: false,
                    tension: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        displayColors: false,
                        callbacks: {
                            title: function() { return null; },
                            label: function(context) { return 'index: ' + context.dataIndex; },
                            afterLabel: function(context) { return 'value: ' + context.parsed.y; }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(255,255,255,0.05)'
                        },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { display: false }
                    }
                }
            }
        });
    }
});

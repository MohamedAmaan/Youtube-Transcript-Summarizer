const btn = document.getElementById("summarize");
btn.addEventListener("click", function() {
    btn.disabled = true;
    btn.innerHTML = "Summarizing...";
    chrome.tabs.query({currentWindow: true, active: true}, function(tabs) {
        var url = tabs[0].url;
        var maxLength = document.getElementById("max_length").value || 150;
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "http://127.0.0.1:5000/summary?url=" + url + "&max_length=" + maxLength, true);
        xhr.onload = function() {
            var text = xhr.responseText;
            const p = document.getElementById("output");
            if (xhr.status === 404) {
                p.innerHTML = "No subtitles available for this video";
            } else {
                p.innerHTML = text;
            }
            btn.disabled = false;
            btn.innerHTML = "Summarize";
        }
        xhr.send();
    });
});
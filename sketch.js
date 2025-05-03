let bodyPose;
let userPose;
let poseData;
let uploadedImg;
let sculpturesDataset = {};

document.getElementById("uploadImage").addEventListener("change", handleImage);

// Load the pose database
fetch("pose_results.json")
  .then((res) => res.json())
  .then((data) => {
    poseData = data;
  });

// Load the sculptures dataset
fetch("sculptures_dataset.csv")
  .then((response) => {
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.text();
  })
  .then((csvText) => {
    // Parse the CSV into an object
    const rows = csvText.split("\n").slice(1); // Skip the header row
    rows.forEach((row) => {
      const columns = row.split(",");
      const objectId = columns[0].trim(); // Object ID
      const title = columns[2]?.trim(); // Title
      const linkResource = columns[5]?.trim(); // Link Resource
      console.log(
        `Parsed Object ID: ${objectId}, Link Resource: ${linkResource}`
      );
      sculpturesDataset[objectId] = { title, linkResource }; // Map Object ID to Title and Link Resource
    });
    console.log("✅ Sculptures dataset loaded");
  })
  .catch((error) => {
    console.error("Error loading sculptures dataset:", error);
  });
// Initialize bodyPose model
bodyPose = ml5.bodyPose(() => {
  console.log("✅ BodyPose model ready");
});

function handleImage(e) {
  const file = e.target.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = () => {
    uploadedImg = img;

    // Display the uploaded image
    const preview = document.getElementById("uploadedImagePreview");
    preview.src = img.src;

    // Process the pose
    processPose(img);
  };
  img.src = URL.createObjectURL(file);
}

function processPose(img) {
  console.log("Processing image:", img);
  console.log("Image dimensions:", img.naturalWidth, img.naturalHeight);

  if (img.naturalWidth === 0 || img.naturalHeight === 0) {
    console.error("Invalid image dimensions.");
    return;
  }

  // Create a p5 image from DOM Image
  let p5Image = new p5.Image(img.width, img.height);
  p5Image.canvas.getContext("2d").drawImage(img, 0, 0);

  bodyPose.detect(p5Image, (poses) => {
    if (poses.length > 0) {
      userPose = poses[0];
      console.log("✅ User pose detected", userPose);

      // Draw pose on the uploaded image canvas
      const canvas = document.getElementById("uploadedImageCanvas");
      drawPoseOnCanvas(canvas, img, userPose);

      findBestMatches();
    } else {
      console.log("❌ No pose detected");
    }
  });
}

function findBestMatches() {
  // Build user vector
  let userVector = keypointsToVector(userPose.keypoints);

  let scored = poseData.map((sculpture) => {
    let sculptureVector = keypointsToVector(sculpture.keypoints);
    let distance = euclideanDistance(userVector, sculptureVector);
    return { ...sculpture, distance };
  });

  // Sort by distance (closest first)
  scored.sort((a, b) => a.distance - b.distance);

  // Show top 3
  let resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "<h3>Matches!</h3>";
  scored.slice(0, 3).forEach((match) => {
    const resultDiv = document.createElement("div");
    resultDiv.style.position = "relative";
    resultDiv.style.paddingTop = "160px";
    resultDiv.style.paddingBottom = "160px";

    // Get the title and link resource from the sculptures dataset
    const sculptureData = sculpturesDataset[match.object_id] || {};
    const title = sculptureData.title || "Unknown Title";
    const linkResource =
      sculptureData.linkResource || "https://www.metmuseum.org/art/collection";

    resultDiv.innerHTML = `
      <img src="${
        match.image_url
      }" class="result-image" style="padding-left: 40px;" />
      <canvas class="result-canvas" style="padding-left: 40px;"></canvas>
      <p style="padding-left: 40px;">
        <a href="${linkResource}" target="_blank">${title}</a> - Distance: ${match.distance.toFixed(
      3
    )}
      </p>
    `;
    resultsDiv.appendChild(resultDiv);

    // Draw pose on the result image canvas
    const img = new Image();
    img.onload = () => {
      const canvas = resultDiv.querySelector("canvas");
      const displayedWidth = img.naturalWidth;
      const displayedHeight = img.naturalHeight;

      // Resize canvas to match the displayed size of the image
      canvas.width = displayedWidth;
      canvas.height = displayedHeight;

      console.log(
        "Drawing pose on canvas with size:",
        canvas.width,
        canvas.height
      );
      console.log("Pose data:", match);
      drawPoseOnCanvas(canvas, img, { keypoints: match.keypoints });
    };
    img.src = match.image_url;
  });
}

function keypointsToVector(keypoints) {
  return keypoints
    .filter((kp) => kp.confidence > 0.2 || kp.visibility > 0.2) // Adjust depending on format
    .map((kp) => [kp.x, kp.y])
    .flat();
}

function euclideanDistance(vec1, vec2) {
  let len = Math.min(vec1.length, vec2.length);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    let diff = vec1[i] - vec2[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function drawPoseOnCanvas(canvas, img, pose) {
  const ctx = canvas.getContext("2d");

  // Debug: Check pose object
  console.log("Pose object:", pose);
  if (!pose || !pose.keypoints || pose.keypoints.length === 0) {
    console.error("Invalid or empty keypoints:", pose);
    return; // Exit if keypoints are invalid
  }

  // Get the displayed size of the image
  const displayedWidth = img.width; // Displayed width (e.g., 400px)
  const displayedHeight = img.height; // Displayed height (scaled proportionally)

  // Resize canvas to match the displayed size of the image
  canvas.width = displayedWidth;
  canvas.height = displayedHeight;

  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Scale the pose coordinates to match the displayed size
  const scaleX = displayedWidth / img.naturalWidth;
  const scaleY = displayedHeight / img.naturalHeight;
  console.log("Scale factors:", scaleX, scaleY);

  // Draw the skeleton connections
  const connections = bodyPose.getSkeleton();
  console.log("Skeleton connections:", connections);
  connections.forEach(([startIdx, endIdx]) => {
    const start = pose.keypoints[startIdx];
    const end = pose.keypoints[endIdx];
    if (start.confidence > 0.2 && end.confidence > 0.2) {
      ctx.strokeStyle = "rgb(209, 253, 88)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(start.x * scaleX, start.y * scaleY);
      ctx.lineTo(end.x * scaleX, end.y * scaleY);
      ctx.stroke();
    }
  });

  // Draw the keypoints
  pose.keypoints.forEach((keypoint) => {
    if (keypoint.confidence > 0.5) {
      const x = keypoint.x * scaleX;
      const y = keypoint.y * scaleY;
      ctx.fillStyle = "rgb(128, 65, 255)";
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
}

// Webcam capture
const webcam = document.getElementById("webcam");
const captureButton = document.getElementById("captureButton");
const uploadedImagePreview = document.getElementById("uploadedImagePreview");

uploadedImagePreview.onload = () => {
  const canvas = document.createElement("canvas");
  canvas.width = uploadedImagePreview.naturalWidth;
  canvas.height = uploadedImagePreview.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(uploadedImagePreview, 0, 0);

  const p5Image = new p5.Image(canvas.width, canvas.height);
  p5Image.canvas.getContext("2d").drawImage(canvas, 0, 0);

  bodyPose.detect(p5Image, (poses) => {
    if (poses.length > 0) {
      userPose = poses[0];
      console.log("✅ User pose detected", userPose);

      const canvas = document.getElementById("uploadedImageCanvas");
      drawPoseOnCanvas(canvas, uploadedImagePreview, userPose);

      findBestMatches();
    } else {
      console.log("❌ No pose detected");
    }
  });
};

// Access the webcam
navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    webcam.srcObject = stream;
    webcam.onloadedmetadata = () => {
      console.log("Webcam dimensions:", webcam.videoWidth, webcam.videoHeight);
    };
  })
  .catch((err) => {
    console.error("Error accessing webcam:", err);
  });

const webcamContainer = document.getElementById("webcamContainer");
const toggleWebcamButton = document.getElementById("toggleWebcamButton");

// Toggle webcam visibility
toggleWebcamButton.addEventListener("click", () => {
  console.log(
    "Current webcamContainer display:",
    webcamContainer.style.display
  );
  if (webcamContainer.style.display === "none") {
    // Show the webcam
    webcamContainer.style.display = "flex";
    webcamContainer.style.flexDirection = "column";
    webcamContainer.style.gap = "16px";
    toggleWebcamButton.textContent = "Hide Webcam";

    // Start the webcam stream
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        console.log("Webcam stream started successfully");
        webcam.srcObject = stream;
        webcam.onloadedmetadata = () => {
          console.log(
            "Webcam dimensions:",
            webcam.videoWidth,
            webcam.videoHeight
          );
        };
      })
      .catch((err) => {
        console.error("Error accessing webcam:", err);
      });
  } else {
    // Hide the webcam
    webcamContainer.style.display = "none";
    toggleWebcamButton.textContent = "Take a photo";

    // Stop the webcam stream
    if (webcam.srcObject) {
      const tracks = webcam.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      webcam.srcObject = null;
    }
  }
});

// Hide webcam after capturing an image
captureButton.addEventListener("click", () => {
  if (webcam.videoWidth === 0 || webcam.videoHeight === 0) {
    console.error("Webcam is not ready.");
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = webcam.videoWidth;
  canvas.height = webcam.videoHeight;
  const ctx = canvas.getContext("2d");

  // Draw the current frame from the video onto the canvas
  ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

  // Convert the canvas to a data URL and set it as the image source
  uploadedImagePreview.src = canvas.toDataURL("image/png");

  // Process the captured image
  processPose(uploadedImagePreview);

  // Show the "Clear" button
  clearButton.style.display = "block";

  // Hide the webcam
  webcamContainer.style.display = "none";
  toggleWebcamButton.textContent = "Take a Photo";

  // Stop the webcam stream
  if (webcam.srcObject) {
    const tracks = webcam.srcObject.getTracks();
    tracks.forEach((track) => track.stop());
    webcam.srcObject = null;
  }
});

// Hide webcam after uploading an image
document.getElementById("uploadImage").addEventListener("change", (e) => {
  handleImage(e);

  // Hide the webcam
  webcamContainer.style.display = "none";
  toggleWebcamButton.textContent = "Take a Photo";

  // Stop the webcam stream
  if (webcam.srcObject) {
    const tracks = webcam.srcObject.getTracks();
    tracks.forEach((track) => track.stop());
    webcam.srcObject = null;
  }
});

const clearButton = document.getElementById("clearButton");

// Show the "Clear" button when an image is uploaded
document.getElementById("uploadImage").addEventListener("change", (e) => {
  handleImage(e);

  // Show the "Clear" button
  clearButton.style.display = "block";

  // Hide the webcam
  webcamContainer.style.display = "none";
  toggleWebcamButton.textContent = "Take a Photo";

  // Stop the webcam stream
  if (webcam.srcObject) {
    const tracks = webcam.srcObject.getTracks();
    tracks.forEach((track) => track.stop());
    webcam.srcObject = null;
  }
});

// Show the "Clear" button when an image is captured from the webcam
captureButton.addEventListener("click", () => {
  if (webcam.videoWidth === 0 || webcam.videoHeight === 0) {
    console.error("Webcam is not ready.");
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = webcam.videoWidth;
  canvas.height = webcam.videoHeight;
  const ctx = canvas.getContext("2d");

  // Draw the current frame from the video onto the canvas
  ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

  // Convert the canvas to a data URL and set it as the image source
  uploadedImagePreview.src = canvas.toDataURL("image/png");

  // Process the captured image
  processPose(uploadedImagePreview);

  // Show the "Clear" button
  clearButton.style.display = "block";

  // Hide the webcam
  webcamContainer.style.display = "none";
  toggleWebcamButton.textContent = "Take a Photo";

  // Stop the webcam stream
  if (webcam.srcObject) {
    const tracks = webcam.srcObject.getTracks();
    tracks.forEach((track) => track.stop());
    webcam.srcObject = null;
  }
});

// Hide the "Clear" button when clicked
clearButton.addEventListener("click", () => {
  // Clear the uploaded image preview
  uploadedImagePreview.src = "";

  // Clear the canvas
  const ctx = uploadedImageCanvas.getContext("2d");
  ctx.clearRect(0, 0, uploadedImageCanvas.width, uploadedImageCanvas.height);

  // Clear only the result divs inside "results"
  const resultsDiv = document.getElementById("results");
  const resultDivs = resultsDiv.querySelectorAll("div");
  resultDivs.forEach((div) => div.remove());

  // Reset the heading to default
  resultsDiv.innerHTML =
    '<h3><span style="color: rgb(142, 86, 255)">✦</span>Results</h3>';

  // Hide the "Clear" button
  clearButton.style.display = "none";

  console.log("Cleared uploaded image and canvas.");
});

const aboutLink = document.getElementById("aboutLink");
const aboutModal = document.getElementById("aboutModal");
const closeButton = document.querySelector(".close-button");

// Show the modal when "About" is clicked
aboutLink.addEventListener("click", (e) => {
  e.preventDefault(); // Prevent default link behavior
  aboutModal.style.display = "flex"; // Show the modal
});

// Hide the modal when the close button is clicked
closeButton.addEventListener("click", () => {
  aboutModal.style.display = "none"; // Hide the modal
});

// Hide the modal when clicking outside the modal content
window.addEventListener("click", (e) => {
  if (e.target === aboutModal) {
    aboutModal.style.display = "none"; // Hide the modal
  }
});

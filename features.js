// Shared feature extraction: used by the live app and the dataset builder,
// so training data and webcam input are processed identically.

export function handVector(lm) {
  const w = lm[0];
  const pts = lm.map((p) => [p.x - w.x, p.y - w.y, p.z - w.z]);
  const scale = Math.max(...pts.map((p) => Math.hypot(p[0], p[1]))) || 1;
  return pts.flatMap((p) => [p[0] / scale, p[1] / scale, p[2] / scale]);
}

// 126-dim vector: up to two hands ordered left-to-right, missing hand = zeros.
export function features(result) {
  if (!result.landmarks || result.landmarks.length === 0) return null;
  const hands = [...result.landmarks].sort((a, b) => a[0].x - b[0].x).slice(0, 2);
  const zeros = new Array(63).fill(0);
  return [...handVector(hands[0]), ...(hands[1] ? handVector(hands[1]) : zeros)];
}

// ASL alphabet features (must match tools/train_asl.py): one hand's keypoints,
// mirrored to a single chirality, plus all 210 pairwise keypoint distances.
export function aslFeatures(lm) {
  const v = handVector(lm);
  const ax = v[15] - v[0], ay = v[16] - v[1];   // wrist -> index MCP (5)
  const bx = v[51] - v[0], by = v[52] - v[1];   // wrist -> pinky MCP (17)
  if (ax * by - ay * bx < 0) for (let i = 0; i < 63; i += 3) v[i] = -v[i];
  const out = v.slice();
  for (let i = 0; i < 21; i++) {
    for (let j = i + 1; j < 21; j++) {
      const dx = v[i * 3] - v[j * 3], dy = v[i * 3 + 1] - v[j * 3 + 1], dz = v[i * 3 + 2] - v[j * 3 + 2];
      out.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
    }
  }
  return out;
}

// Pretrained samples are stored as 63-dim (one hand); pad to the full width.
export function padVector(v) {
  return v.length >= 126 ? v : [...v, ...new Array(126 - v.length).fill(0)];
}

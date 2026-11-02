// Decorative wave drawing only: no telemetry, remote requests, or service connection.
const waves = document.getElementById('waves');
if (waves) {
  for (let line = 0; line < 19; line++) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    let d = '';
    for (let x = 0; x <= 600; x += 4) {
      const envelope = Math.exp(-Math.pow((x - 300) / 145, 2));
      const y = 225 + line * 6 - Math.sin(x / 55 + line * .13) * (30 + line * 2) * envelope;
      d += `${x ? 'L' : 'M'}${x} ${y.toFixed(2)} `;
    }
    path.setAttribute('d', d);
    path.setAttribute('opacity', String(.25 + line * .035));
    waves.appendChild(path);
  }
}

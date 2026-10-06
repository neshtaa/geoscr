const puppeteer = require('puppeteer');

(async () => {
  console.log('[*] Connecting to X11 DISPLAY=:0 and launching dedicated calibration window...');
  const browser = await puppeteer.launch({
    headless: false,
    defaultViewport: null,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--window-size=1440,920',
      '--window-position=50,30'
    ]
  });

  const pages = await browser.pages();
  const page = pages[0] || await browser.newPage();
  await page.setViewport({ width: 1400, height: 880 });

  console.log('[*] Navigating to http://localhost:8080/calibration ...');
  await page.goto('http://localhost:8080/calibration', { waitUntil: 'networkidle0' });
  console.log('[+] Calibration Dashboard successfully rendered on user monitor (DISPLAY=:0)!');

  // Short pause so user sees the initial round loaded
  await new Promise(r => setTimeout(r, 2000));

  // Trigger automated calibration benchmark across all 10 rounds
  console.log('[*] Triggering automated calibration benchmark across all 10 test rounds...');
  await page.click('#run-all-btn');

  // Monitor benchmark progress
  let finished = false;
  for (let step = 0; step < 40; step++) {
    await new Promise(r => setTimeout(r, 1200));
    const status = await page.evaluate(() => {
      const stats = document.getElementById('progress-stats');
      const label = document.getElementById('progress-label');
      const btn = document.getElementById('run-all-btn');
      const counter = document.getElementById('accuracy-counter');
      return {
        stats: stats ? stats.innerText : '',
        label: label ? label.innerText : '',
        btnText: btn ? btn.innerText : '',
        btnDisabled: btn ? btn.disabled : false,
        accuracy: counter ? counter.innerText : ''
      };
    });

    console.log(`[Calibration Status] ${status.label} | ${status.stats} | ${status.accuracy}`);

    if (!status.btnDisabled && status.btnText.includes('Complete')) {
      finished = true;
      console.log('[+] All 10 rounds successfully calibrated and verified on screen!');
      break;
    }
  }

  // Take screenshot for documentation
  await page.screenshot({ path: 'scratch/calibration_dashboard_verified.png' });
  console.log('[+] Saved verification screenshot to scratch/calibration_dashboard_verified.png');

  // Keep window open indefinitely on user desktop
  console.log('[*] Calibration window remains actively displayed on screen for user inspection.');
})();

const puppeteer = require('puppeteer');

const code = process.argv[2] || '';
console.log('Opening GitHub device authorization window for code:', code);

(async () => {
  try {
    const browser = await puppeteer.launch({
      headless: false,
      defaultViewport: null,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--window-size=1200,850',
        '--window-position=100,50'
      ]
    });

    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36');
    await page.goto('https://github.com/login/device', { waitUntil: 'networkidle2' });

    console.log('[+] GitHub authorization page opened successfully on DISPLAY=:0');

    if (code) {
      try {
        await page.waitForSelector('#user_code, input[name="user_code"]', { timeout: 6000 });
        const input = await page.$('#user_code, input[name="user_code"]');
        if (input) {
          await input.type(code);
          console.log('[+] One-time code automatically populated in input field!');
        }
      } catch (err) {
        console.log('[*] Login required before code entry, or user code selector changed.');
      }
    }

    // Keep browser alive until user completes authorization or closes window
    browser.on('disconnected', () => {
      console.log('Browser window closed by user.');
      process.exit(0);
    });

  } catch (err) {
    console.error('Error opening browser:', err);
    process.exit(1);
  }
})();

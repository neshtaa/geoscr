/**
 * Interactive GeoGuessr Login and Clues Extractor
 * Launches a visible Chrome window on the user's desktop display.
 * Captures `_ncfa` cookie upon login, saves it to data/session_cookie.txt,
 * and automatically dumps the complete GeoGuessr clues database.
 */

const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

puppeteer.use(StealthPlugin());

const DISPLAY = process.env.DISPLAY || ':0';

async function main() {
    console.log(`\n======================================================`);
    console.log(`🚀 ЗАПУСК БРАУЗЕРА ДЛЯ ВХОДУ В GEOGUESSR`);
    console.log(`🖥️  Дисплей: ${DISPLAY}`);
    console.log(`📌 Відкрито сторінку: https://www.geoguessr.com/signin`);
    console.log(`👉 Будь ласка, увійдіть у свій акаунт (Google, пошта тощо).`);
    console.log(`⌛ Очікую авторизації та появи сесійного cookie (_ncfa)...`);
    console.log(`======================================================\n`);

    const browser = await puppeteer.launch({
        headless: false,
        executablePath: '/home/yurka/.cache/puppeteer/chrome/linux-148.0.7778.97/chrome-linux64/chrome',
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--window-size=1280,920',
            '--start-maximized'
        ],
        defaultViewport: null
    });

    const pages = await browser.pages();
    const page = pages.length > 0 ? pages[0] : await browser.newPage();

    try {
        await page.goto('https://www.geoguessr.com/signin', { waitUntil: 'domcontentloaded' });
    } catch (e) {
        console.log(`[*] Навігація: ${e.message}`);
    }

    let capturedCookie = null;
    const timeoutMs = 15 * 60 * 1000; // 15 хвилин
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
        // Перевіряємо, чи браузер ще відкритий
        if (!browser.isConnected()) {
            console.log('[!] Браузер закрито користувачем.');
            break;
        }

        try {
            const context = browser.defaultBrowserContext();
            const cookies = await context.cookies();
            const ncfa = cookies.find(c => c.name === '_ncfa');
            if (ncfa && ncfa.value) {
                capturedCookie = ncfa.value;
                console.log(`\n🎉 [УСПІХ] Авторизація зафіксована! Отримано _ncfa токен.`);
                break;
            }
        } catch (err) {
            // Ігноруємо помилки переходу сторінок
        }

        await new Promise(r => setTimeout(r, 1500));
    }

    if (!capturedCookie) {
        console.error('\n[!] Сесійний cookie не знайдено або час очікування вичерпано.');
        if (browser.isConnected()) await browser.close();
        process.exit(1);
    }

    // Зберігаємо cookie в data/session_cookie.txt
    const cookieDir = path.join(__dirname, 'data');
    if (!fs.existsSync(cookieDir)) fs.mkdirSync(cookieDir, { recursive: true });
    const cookieFile = path.join(cookieDir, 'session_cookie.txt');
    fs.writeFileSync(cookieFile, capturedCookie.trim(), 'utf-8');
    console.log(`💾 Токен сесії збережено в: ${cookieFile}`);

    // Чекаємо 2 секунди, щоб сесія повністю збереглася на боці GeoGuessr
    await new Promise(r => setTimeout(r, 2000));
    if (browser.isConnected()) {
        console.log('🔒 Закриваємо вікно браузера...');
        await browser.close();
    }

    console.log('\n📥 Запуск автоматичного вивантаження бази підказок GeoGuessr...');
    const pythonScript = path.join(__dirname, 'scraper_geoguessr_clues.py');
    const proc = spawn('python3', [pythonScript, capturedCookie], { stdio: 'inherit' });

    proc.on('close', (code) => {
        if (code === 0) {
            console.log('\n✅ [ГОТОВО] Повну базу підказок GeoGuessr успішно спарсено та збережено!');
        } else {
            console.error(`\n[!] Парсер завершився з кодом ${code}`);
        }
        process.exit(code);
    });
}

main().catch(err => {
    console.error('[!] Помилка:', err);
    process.exit(1);
});

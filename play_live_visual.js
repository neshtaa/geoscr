const puppeteer = require('puppeteer');
const fs = require('fs');

const CHALLENGES = [
  { token: '0h8xlPiMeF8rRTQw', name: 'A Souvlaki World (Global Handpicked World Map)' },
  { token: 'axCmrmKtjX0klf9n', name: 'A Souvlaki World (Global Handpicked World Map)' },
  { token: '0IQtDydj8reRK6Zk', name: 'A Souvlaki World (Global Handpicked World Map)' }
];



function getCookie() {
  const cookiePath = 'data/session_cookie.txt';
  if (fs.existsSync(cookiePath)) {
    return fs.readFileSync(cookiePath, 'utf8').trim();
  }
  return process.env.GEOGUESSR_COOKIE || '';
}

async function apiRequest(url, method = 'GET', data = null, cookie = '') {
  const headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Origin': 'https://www.geoguessr.com',
    'Referer': 'https://www.geoguessr.com/'
  };
  if (cookie) {
    headers['Cookie'] = `_ncfa=${cookie}`;
  }
  const opts = { method, headers };
  if (data) {
    headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(data);
  }
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${errText.slice(0, 150)}`);
  }
  return await resp.json();
}

async function getHeading(page) {
  try {
    return await page.evaluate(() => {
      const labels = Array.from(document.querySelectorAll("[class*=\"latitudeLabel\"]"));
      const indicator = document.querySelector("[class*=\"topIndicator\"], [class*=\"bottomIndicator\"]");
      const container = document.querySelector("[class*=\"panorama-compass_compass\"]");
      const targetX = indicator ? indicator.getBoundingClientRect().left : (container ? container.getBoundingClientRect().left + container.getBoundingClientRect().width / 2 : 0);
      
      let closest = null, minD = 9999;
      labels.forEach(l => {
        const text = (l.innerText || "").trim().toUpperCase();
        if (!text) return;
        const b = l.getBoundingClientRect();
        const d = Math.abs((b.left + b.width / 2) - targetX);
        if (d < minD) { minD = d; closest = text; }
      });
      return closest;
    });
  } catch(e) {
    return null;
  }
}

async function alignCameraTo(page, targetHeading) {
  const canvas = await page.$("[data-qa='panorama-canvas']") || await page.$("canvas.widget-scene-canvas") || await page.$("canvas");
  if (!canvas) return false;
  const box = await canvas.boundingBox();
  if (!box) return false;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

  for (let i = 0; i < 20; i++) {
    const h = await getHeading(page);
    if (!h) {
      await new Promise(r => setTimeout(r, 250));
      continue;
    }
    if (h === targetHeading) {
      console.log(`[+] Camera successfully locked onto heading ${targetHeading}`);
      return true;
    }
    const curIdx = order.indexOf(h);
    const tgtIdx = order.indexOf(targetHeading);
    if (curIdx === -1 || tgtIdx === -1) {
      await new Promise(r => setTimeout(r, 200));
      continue;
    }
    const diff = (tgtIdx - curIdx + 8) % 8;
    const isAdjacent = (diff === 1 || diff === 7);
    const step = isAdjacent ? 130 : 250;
    const dx = diff <= 4 ? -step : step;

    await page.mouse.move(cx, cy);
    await page.mouse.down();
    await page.mouse.move(cx + dx, cy, { steps: 5 });
    await page.mouse.up();
    await new Promise(r => setTimeout(r, 220));
  }
  return false;
}

async function injectOrUpdateHud(page, state) {
  try {
    await page.evaluate((s) => {
      let hud = document.getElementById('geoscript-live-hud');
      if (!hud) {
        hud = document.createElement('div');
        hud.id = 'geoscript-live-hud';
        hud.style.cssText = `
          position: fixed;
          top: 20px;
          right: 20px;
          width: 370px;
          max-height: 88vh;
          overflow-y: auto;
          background: rgba(15, 23, 42, 0.95);
          backdrop-filter: blur(14px);
          border: 2px solid #10b981;
          border-radius: 14px;
          padding: 16px;
          color: #f8fafc;
          font-family: 'Inter', system-ui, -apple-system, sans-serif;
          font-size: 13px;
          z-index: 99999999;
          box-shadow: 0 16px 40px rgba(0,0,0,0.7);
          transition: all 0.3s ease;
        `;
        document.body.appendChild(hud);
      }

      const altsHtml = (s.alts || []).map(a => 
        `<span style="background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:3px 8px;border-radius:6px;margin:2px;font-size:11px;display:inline-block;">${a}</span>`
      ).join('');

      const cluesHtml = (s.clues || []).map(c => 
        `<span style="background:#0f172a;border:1px solid #0284c7;color:#38bdf8;padding:3px 8px;border-radius:6px;margin:2px;font-size:11px;display:inline-block;">🔍 ${c}</span>`
      ).join('');

      const officialCluesHtml = (s.officialClues || []).map(c => `
        <div style="background:#1e293b;border-left:3px solid #10b981;border-radius:6px;padding:8px 10px;margin-top:6px;">
          <div style="font-weight:700;color:#f8fafc;font-size:12px;">${c.title} <span style="font-size:10px;color:#10b981;text-transform:uppercase;">[${c.category || 'clue'}]</span></div>
          <div style="color:#94a3b8;font-size:11px;margin-top:3px;line-height:1.35;">${c.description}</div>
        </div>
      `).join('');

      hud.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #334155;padding-bottom:10px;margin-bottom:10px;">
          <div style="font-weight:800;font-size:14px;color:#10b981;display:flex;align-items:center;gap:6px;">
            <span>⚡ Deterministic Clue Engine (0% AI)</span>
          </div>
          <span style="background:#10b981;color:#0b1120;padding:3px 10px;border-radius:12px;font-weight:800;font-size:11px;">${s.roundText || 'Live'}</span>
        </div>

        <div style="color:#94a3b8;font-size:12px;margin-bottom:10px;display:flex;align-items:center;gap:6px;">
          <span>${s.statusIcon || '⏳'}</span> <span>${s.statusText || 'Processing...'}</span>
        </div>

        ${s.prediction ? `
          <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #10b981;border-radius:10px;padding:12px;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:18px;font-weight:800;color:#fff;">${s.prediction.country} (${s.prediction.country_code})</span>
              <span style="background:#10b981;color:#0b1120;padding:2px 8px;border-radius:12px;font-weight:800;font-size:12px;">${s.prediction.confidence_percent}%</span>
            </div>
            <div style="color:#f59e0b;font-size:12px;font-weight:600;margin-top:4px;">📍 ${s.prediction.region || 'General Region'}</div>
            ${s.prediction.gps_estimate ? `<div style="color:#94a3b8;font-size:11px;margin-top:2px;">Target Coordinates: ${s.prediction.gps_estimate.lat}, ${s.prediction.gps_estimate.lng}</div>` : ''}
          </div>
        ` : ''}

        ${s.evaluation ? `
          <div style="background:${s.evaluation.is_match ? 'rgba(6, 78, 59, 0.9)' : 'rgba(153, 27, 27, 0.9)'};border:2px solid ${s.evaluation.is_match ? '#34d399' : '#f87171'};border-radius:10px;padding:10px 12px;margin-bottom:10px;">
            <div style="font-weight:900;font-size:13px;color:#fff;display:flex;align-items:center;gap:6px;">
              <span>${s.evaluation.is_match ? '✅ VERIFIED ACCURACY MATCH' : '❌ REGION MISMATCH'}</span>
            </div>
            <div style="font-size:12px;color:#fff;margin-top:4px;">
              <strong>True Location:</strong> ${s.evaluation.true_country} (${s.evaluation.true_code}) ${s.evaluation.true_state ? `[${s.evaluation.true_state}]` : ''}
            </div>
            <div style="font-size:12px;color:#fff;margin-top:2px;">
              <strong>Script Predicted:</strong> ${s.prediction.country} (${s.prediction.country_code})
            </div>
            <div style="font-size:11px;color:${s.evaluation.is_match ? '#a7f3d0' : '#fecaca'};margin-top:4px;font-style:italic;">
              Result: ${s.evaluation.match_type}
            </div>
          </div>
        ` : ''}

        ${altsHtml ? `<div style="margin-bottom:10px;"><div style="font-size:11px;color:#94a3b8;font-weight:600;margin-bottom:4px;">ALTERNATIVE CANDIDATES:</div>${altsHtml}</div>` : ''}
        ${cluesHtml ? `<div style="margin-bottom:10px;"><div style="font-size:11px;color:#94a3b8;font-weight:600;margin-bottom:4px;">EXTRACTED VISUAL CUES:</div>${cluesHtml}</div>` : ''}
        ${officialCluesHtml ? `<div style="margin-bottom:10px;"><div style="font-size:11px;color:#94a3b8;font-weight:600;margin-bottom:4px;">MATCHED GEOGUESSR DATABASE CLUES:</div>${officialCluesHtml}</div>` : ''}

        ${s.roundResult ? `
          <div style="background:#064e3b;border:1px solid #10b981;border-radius:10px;padding:12px;margin-top:10px;">
            <div style="font-weight:800;color:#6ee7b7;font-size:13px;display:flex;justify-content:space-between;">
              <span>🏆 OFFICIAL SERVER SCORE:</span>
              <span>+${s.roundResult.score} pts</span>
            </div>
            <div style="font-size:14px;color:#fff;margin-top:4px;">Distance Error: <strong>${s.roundResult.distance} km</strong></div>
            <div style="font-size:11px;color:#a7f3d0;margin-top:2px;">True Coordinates: lat=${s.roundResult.trueLat}, lng=${s.roundResult.trueLng}</div>
            <div style="border-top:1px solid #059669;margin-top:6px;padding-top:6px;font-size:13px;font-weight:800;color:#fde047;">
              Total Match Score: ${s.roundResult.totalScore} / ${s.roundResult.maxScore} pts
            </div>
          </div>
        ` : ''}
      `;
    }, state);
  } catch (err) {
    // If navigation happens, ignore transient eval error
  }
}

async function dismissModals(page) {
  try {
    for (let attempt = 0; attempt < 3; attempt++) {
      const clicked = await page.evaluate(() => {
        let didClick = false;
        const acceptBtn = document.querySelector('#onetrust-accept-btn-handler, [data-qa="accept-cookies"]');
        if (acceptBtn && acceptBtn.offsetParent !== null) {
          acceptBtn.click();
          didClick = true;
        }

        const closeBtn = document.querySelector('[data-qa="close-round-result"], [data-qa="play-next-round"], [data-qa="close-modal"], [aria-label*="Close" i]');
        if (closeBtn && closeBtn.offsetParent !== null) {
          closeBtn.click();
          didClick = true;
        }

        const btns = Array.from(document.querySelectorAll('button, a, [role="button"]'));
        for (const b of btns) {
          if (b.offsetParent === null) continue;
          const text = (b.innerText || '').trim().toLowerCase();
          if (
            text === 'next' ||
            text === 'play' ||
            text === 'start game' ||
            text === 'play next round' ||
            text === 'next round' ||
            text === 'continue' ||
            text === 'resume' ||
            text.includes('start exploring')
          ) {
            b.click();
            didClick = true;
            break;
          }
        }
        return didClick;
      });
      if (clicked) {
        await new Promise(r => setTimeout(r, 1200));
      } else {
        break;
      }
    }
  } catch(e) {}
}

async function playChallenge(challengeIndex, browser, cookie) {
  const challenge = CHALLENGES[challengeIndex % CHALLENGES.length];
  console.log('\n' + '='.repeat(70));
  console.log(`🎮 LAUNCHING LIVE MATCH ${challengeIndex + 1} / ${CHALLENGES.length}`);
  console.log(`🌍 Map: ${challenge.name} (Token: ${challenge.token})`);
  console.log('='.repeat(70));

  // Initialize challenge session
  const game = await apiRequest(`https://www.geoguessr.com/api/v3/challenges/${challenge.token}`, 'POST', {}, cookie);
  const gameToken = game.token;
  let gameState = game;
  try {
    gameState = await apiRequest(`https://www.geoguessr.com/api/v3/games/${gameToken}`, 'GET', null, cookie);
  } catch(e) {}

  const totalRounds = gameState.roundCount || game.roundCount || 5;
  const existingGuesses = (gameState.player && gameState.player.guesses) ? gameState.player.guesses.length : 0;
  console.log(`[+] Live Game Session Active! Game Token: ${gameToken}, Existing guesses: ${existingGuesses}/${totalRounds}`);

  if (existingGuesses >= totalRounds || gameState.state === 'finished') {
    console.log(`[!] Game ${gameToken} is already finished. Skipping to next challenge.`);
    return null;
  }

  const startRound = existingGuesses + 1;
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  await page.setCookie({ name: '_ncfa', value: cookie, domain: '.geoguessr.com' });

  // Navigate to live game
  console.log(`[*] Loading game page in visible browser window...`);
  await page.goto(`https://www.geoguessr.com/game/${gameToken}`, { waitUntil: 'networkidle2', timeout: 45000 });
  await new Promise(r => setTimeout(r, 2000));
  await dismissModals(page);
  await new Promise(r => setTimeout(r, 1500));

  let totalScore = 0;
  if (game.player && game.player.guesses) {
    totalScore = game.player.guesses.reduce((sum, g) => sum + parseInt(g.roundScoreInPoints || 0), 0);
  }
  const matchScores = [];

  for (let rNum = startRound; rNum <= totalRounds; rNum++) {
    console.log(`\n────────────────────────────────────────────────────────────────────`);
    console.log(`📍 ROUND ${rNum} / ${totalRounds}`);
    console.log(`────────────────────────────────────────────────────────────────────`);

    // Dismiss previous modal or cookie banner if present
    await dismissModals(page);
    await new Promise(r => setTimeout(r, 1000));

    // Find canvas
    await page.waitForSelector("[data-qa='panorama-canvas'], canvas.widget-scene-canvas", { timeout: 15000 });
    const canvasEl = await page.$("[data-qa='panorama-canvas']") || await page.$("canvas.widget-scene-canvas");
    await new Promise(r => setTimeout(r, 2200));

    // 1. Compass Lock North
    await injectOrUpdateHud(page, {
      roundText: `Round ${rNum} / ${totalRounds}`,
      statusIcon: '🧭',
      statusText: 'Aligning camera due North via compass...'
    });

    await alignCameraTo(page, 'N');
    await dismissModals(page);
    await new Promise(r => setTimeout(r, 600));

    // Capture clean North screenshot
    let imgNorthBuffer = await canvasEl.screenshot({ type: 'jpeg', quality: 90 });
    let hasModal = await page.evaluate(() => !!document.querySelector('[role="dialog"], [class*="modal" i]'));
    if (hasModal) {
      await dismissModals(page);
      await new Promise(r => setTimeout(r, 1000));
      imgNorthBuffer = await canvasEl.screenshot({ type: 'jpeg', quality: 90 });
    }
    const b64North = imgNorthBuffer.toString('base64');

    // 2. Smoothly rotate 180° to face South
    await injectOrUpdateHud(page, {
      roundText: `Round ${rNum} / ${totalRounds}`,
      statusIcon: '🔄',
      statusText: 'Rotating camera 180° to inspect Southern sky & road markings...'
    });

    await alignCameraTo(page, 'S');
    await dismissModals(page);
    await new Promise(r => setTimeout(r, 600));

    // Capture clean South screenshot
    let imgSouthBuffer = await canvasEl.screenshot({ type: 'jpeg', quality: 90 });
    hasModal = await page.evaluate(() => !!document.querySelector('[role="dialog"], [class*="modal" i]'));
    if (hasModal) {
      await dismissModals(page);
      await new Promise(r => setTimeout(r, 1000));
      imgSouthBuffer = await canvasEl.screenshot({ type: 'jpeg', quality: 90 });
    }
    const b64South = imgSouthBuffer.toString('base64');

    await injectOrUpdateHud(page, {
      roundText: `Round ${rNum} / ${totalRounds}`,
      statusIcon: '🧠',
      statusText: 'Analyzing solar azimuth & matching GeoGuessr knowledge base...'
    });

    // Request prediction with dual perspective (North & South)
    const predictResp = await fetch('http://localhost:8080/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        image_b64: b64North,
        image_south_b64: b64South
      })
    });
    const predictionData = await predictResp.json();


    const top = predictionData.top_prediction;
    const alts = (predictionData.alternative_candidates || []).map(a => `${a.country} (${a.confidence_percent}%)`);
    const clues = predictionData.identified_clues || [];
    const officialClues = top.official_clues || [];

    console.log(`[+] Prediction: ${top.country} (${top.country_code}) [${top.confidence_percent}% confidence]`);
    console.log(`    Region:     ${top.region}`);
    console.log(`    Signals:    ${clues.join(', ')}`);
    console.log(`    Target GPS: lat=${top.gps_estimate.lat}, lng=${top.gps_estimate.lng}`);

    await injectOrUpdateHud(page, {
      roundText: `Round ${rNum} / ${totalRounds}`,
      statusIcon: '🎯',
      statusText: 'Deterministic Geolocation deduced! Submitting official guess...',
      prediction: top,
      alts: alts.slice(0, 3),
      clues: clues,
      officialClues: officialClues.slice(0, 2)
    });

    // Brief pause so user can view the on-screen prediction
    await new Promise(r => setTimeout(r, 2500));

    // Submit guess to GeoGuessr
    let guessData;
    try {
      guessData = await apiRequest(`https://www.geoguessr.com/api/v3/games/${gameToken}`, 'POST', {
        token: gameToken,
        lat: top.gps_estimate.lat,
        lng: top.gps_estimate.lng,
        timedOut: false
      }, cookie);
    } catch (err) {
      if (err.message && err.message.includes('AlreadyGuessed')) {
        console.log('[!] Round already registered by server, retrieving current game state...');
        guessData = await apiRequest(`https://www.geoguessr.com/api/v3/games/${gameToken}`, 'GET', null, cookie);
      } else {
        throw err;
      }
    }

    const player = guessData.player || {};
    const lastGuess = (player.guesses && player.guesses.length) ? player.guesses[player.guesses.length - 1] : {};
    const roundPts = parseInt(lastGuess.roundScoreInPoints || 0);
    const distKm = Math.round((parseFloat(lastGuess.distanceInMeters || 0) / 1000) * 10) / 10;
    totalScore += roundPts;

    const currentRoundIndex = (player.guesses && player.guesses.length) ? player.guesses.length - 1 : rNum - 1;
    const roundInfo = (guessData.rounds && guessData.rounds.length > currentRoundIndex) ? guessData.rounds[currentRoundIndex] : (guessData.rounds && guessData.rounds.length >= rNum ? guessData.rounds[rNum - 1] : {});
    const trueLat = roundInfo.lat;
    const trueLng = roundInfo.lng;

    // Authoritative Ground Truth Evaluation
    let evalRes = { is_match: false, match_type: "UNKNOWN", true_code: "??", true_country: "Unknown" };
    try {
      const evalResp = await fetch('http://localhost:8080/api/evaluate_round', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: trueLat,
          lng: trueLng,
          pred_code: top.country_code
        })
      });
      evalRes = await evalResp.json();
    } catch(e) {
      console.error('[!] Evaluation error:', e);
    }

    const matchSymbol = evalRes.is_match ? '✅' : '❌';
    console.log(`\n${matchSymbol} [ACCURACY CHECK] ${evalRes.match_type}`);
    console.log(`   Ground Truth:     ${evalRes.true_country} (${evalRes.true_code}) [${evalRes.true_state || 'N/A'}]`);
    console.log(`   Script Predicted: ${top.country} (${top.country_code})`);
    console.log(`🏆 Round Score:    ${roundPts.toLocaleString()} / 5,000 points`);
    console.log(`   Distance Error: ${distKm.toLocaleString()} km`);
    console.log(`   True Location:  lat=${trueLat}, lng=${trueLng}`);
    console.log(`   Total Score:    ${totalScore.toLocaleString()} / ${rNum * 5000} pts`);

    matchScores.push({
      round: rNum,
      predicted: `${top.country} (${top.country_code})`,
      true_country: `${evalRes.true_country} (${evalRes.true_code})`,
      is_match: evalRes.is_match,
      match_type: evalRes.match_type,
      score: roundPts,
      distance: distKm
    });

    // Save screenshots for offline audit
    try {
      if (!fs.existsSync('scratch')) fs.mkdirSync('scratch', { recursive: true });
      fs.writeFileSync(`scratch/live_${gameToken}_r${rNum}_north.jpg`, imgNorthBuffer);
      fs.writeFileSync(`scratch/live_${gameToken}_r${rNum}_south.jpg`, imgSouthBuffer);
    } catch(e) {}

    // Reload page so the user visually sees GeoGuessr's animated result map and score
    try {
      await page.goto(`https://www.geoguessr.com/game/${gameToken}`, { waitUntil: 'networkidle2', timeout: 35000 });
      await new Promise(r => setTimeout(r, 1500));
    } catch(e) {}

    // Update HUD with final round results on top of the GeoGuessr result map
    await injectOrUpdateHud(page, {
      roundText: `Round ${rNum} / ${totalRounds}`,
      statusIcon: evalRes.is_match ? '✅' : '❌',
      statusText: evalRes.is_match ? `VERIFIED: ${evalRes.match_type}` : `MISMATCH: ${evalRes.match_type}`,
      prediction: top,
      evaluation: evalRes,
      alts: alts.slice(0, 3),
      clues: clues,
      officialClues: officialClues.slice(0, 2),
      roundResult: {
        score: roundPts.toLocaleString(),
        distance: distKm.toLocaleString(),
        trueLat,
        trueLng,
        totalScore: totalScore.toLocaleString(),
        maxScore: (rNum * 5000).toLocaleString()
      }
    });

    // 4.5 seconds for user to inspect the result on screen before advancing
    await new Promise(r => setTimeout(r, 4500));

    // Click Next Round button on screen if not the last round
    if (rNum < totalRounds) {
      try {
        const clicked = await page.evaluate(() => {
          const btns = Array.from(document.querySelectorAll('button, a'));
          const btn = btns.find(b => 
            (b.innerText && (b.innerText.toLowerCase().includes('next') || b.innerText.toLowerCase().includes('play next') || b.innerText.toLowerCase().includes('continue'))) ||
            b.getAttribute('data-qa') === 'play-next-round' || 
            b.getAttribute('data-qa') === 'close-round-result'
          );
          if (btn) {
            btn.click();
            return true;
          }
          return false;
        });
        if (!clicked) {
          const nextBtn = await page.$("[data-qa='close-round-result'], [data-qa='play-next-round'], button[class*='nextButton']");
          if (nextBtn) await nextBtn.click();
        }
        await new Promise(r => setTimeout(r, 2500));

        // Ensure next round canvas is ready; if still showing previous modal, reload game URL
        const hasCanvas = await page.$("[data-qa='panorama-canvas'], canvas.widget-scene-canvas");
        if (!hasCanvas) {
          await page.goto(`https://www.geoguessr.com/game/${gameToken}`, { waitUntil: 'networkidle2', timeout: 35000 });
          await new Promise(r => setTimeout(r, 1500));
        }
      } catch(e) {}
    }
  }

  await page.close();

  console.log('\n' + '='.repeat(70));
  console.log(`📊 FINAL MATCH SUMMARY: ${challenge.name}`);
  console.log(`• Total Score:   ${totalScore.toLocaleString()} / 25,000 points`);
  const avgDist = Math.round((matchScores.reduce((acc, r) => acc + r.distance, 0) / matchScores.length) * 10) / 10;
  console.log(`• Average Distance Error: ${avgDist.toLocaleString()} km`);

  const matchCount = matchScores.filter(r => r.is_match).length;
  const matchRate = Math.round((matchCount / matchScores.length) * 100);
  console.log(`• Verified Country / Neighbor Matches: ${matchCount} / ${matchScores.length} (${matchRate}%)`);
  console.log('\nRound Breakdown:');
  matchScores.forEach(r => {
    const sym = r.is_match ? '✅' : '❌';
    console.log(`  ${sym} Round ${r.round}: Pred: ${r.predicted.padEnd(20)} | True: ${r.true_country.padEnd(22)} | Score: ${r.score.toString().padStart(5)} pts | ${r.distance} km | [${r.match_type}]`);
  });
  console.log('='.repeat(70));
  return { totalScore, avgDist, matchCount, totalRounds: matchScores.length, matchScores };
}

async function main() {
  const cookie = getCookie();
  if (!cookie) {
    console.error('[!] No cookie found in data/session_cookie.txt');
    process.exit(1);
  }

  console.log('[*] Launching graphical browser on desktop DISPLAY=' + (process.env.DISPLAY || ':0'));
  const browser = await puppeteer.launch({
    headless: false, // HEADED MODE: Opens visible browser window on user's screen
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--window-size=1280,820',
      '--window-position=60,60',
      '--disable-infobars'
    ]
  });

  try {
    const singleIndex = process.argv[2] !== undefined ? parseInt(process.argv[2]) : null;
    const toPlay = (singleIndex !== null && !isNaN(singleIndex)) ? [singleIndex] : CHALLENGES.map((_, i) => i);

    for (let i = 0; i < toPlay.length; i++) {
      const cIdx = toPlay[i];
      await playChallenge(cIdx, browser, cookie);
      if (i < toPlay.length - 1) {
        console.log('\n[*] Next challenge starting in 5 seconds...');
        await new Promise(r => setTimeout(r, 5000));
      }
    }

  } catch (err) {
    console.error('[!] Match error:', err);
  } finally {
    console.log('[*] Finished playing matches. Leaving window open for 10 seconds for user inspection...');
    await new Promise(r => setTimeout(r, 10000));
    await browser.close();
    console.log('[+] Browser closed.');
  }
}

main();

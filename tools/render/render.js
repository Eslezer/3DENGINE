// node render.js <model.glb> <out_dir> [views.json]  (run `npm install` here first; CLAY=1 renders untextured geometry)
// Each view: y/x = target as a fraction of the bounding box, dist = camera distance as a
// fraction of model height, azim 0 = front (+Z), elev in degrees.
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
const [glb, outDir, viewsJson = path.join(__dirname, 'views.json')] = process.argv.slice(2);
const views = JSON.parse(fs.readFileSync(viewsJson, 'utf8'));
(async () => {
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage();
  page.on('console', m => console.log('page:', m.text()));
  page.on('pageerror', e => console.log('pageerror:', e.message));
  await page.route('http://r.local/**', route => {
    const p = new URL(route.request().url()).pathname;
    const file = p === '/model.glb' ? glb : path.join(__dirname, p === '/' ? 'page.html' : p);
    const type = file.endsWith('.js') ? 'text/javascript' : file.endsWith('.html') ? 'text/html' : 'application/octet-stream';
    fs.existsSync(file) ? route.fulfill({ body: fs.readFileSync(file), contentType: type }) : route.fulfill({ status: 404 });
  });
  await page.goto('http://r.local/' + (process.env.CLAY ? '?clay' : ''));
  await page.waitForFunction('window.ready === true', null, { timeout: 120000 });
  fs.mkdirSync(outDir, { recursive: true });
  for (const v of views) {
    const url = await page.evaluate(v => window.renderView(v), v);
    fs.writeFileSync(path.join(outDir, v.name + '.png'), Buffer.from(url.split(',')[1], 'base64'));
    console.log(v.name);
  }
  await browser.close();
})();

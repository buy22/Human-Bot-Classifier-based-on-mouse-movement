const puppeteer = require('puppeteer');
const ghostCursor = require('ghost-cursor');
const fs = require('fs');

(async () => {
  const browser = await puppeteer.launch({headless: true});
  const page = await browser.newPage();
  const from = {x: 0, y: 0};
  const to = {x: 1001, y: 520};
  const sampleSize = 1000;
  const paths = [];

  for (let i = 0; i < sampleSize; i++) {
    let positions = ghostCursor.path(from, to);
    let timestamp = Date.now();
    let startTime = Date.now();

    /* if you want the timestamps, move the mouse */
    let path = [];
    await page.mouse.move(positions[0].x, positions[0].y);
    for (j in positions) {
      let v = positions[j];
      await page.mouse.move(v.x, v.y);
      timestamp = Date.now() - startTime;
      path.push({...v, t: timestamp});
    }
    console.log(path);
    console.log(i);
    paths.push(path);

    /* if dont want t */
  }
  fs.writeFile("samples.json", JSON.stringify(paths), (err) => {
    console.log(err);
  });
})();
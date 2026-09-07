// Run with Node and an existing Playwright installation; no server is needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

const source = path.resolve(__dirname, '../../docs/examples/19_4_3_rbe_joint_opening_arch.html');
const output = path.resolve(__dirname, '../../build/viewer-qa');
const html = fs.readFileSync(source, 'utf8');
const jsonPattern = /(<script id="opening-data" type="application\/json">)([\s\S]*?)(<\/script>)/;
const data = JSON.parse(html.match(jsonPattern)[2]);
const errors = [], externalRequests = [];
const active = page => page.locator('[data-active-case]').getAttribute('data-active-case');
const locked = page => page.locator('[data-locked-case]').getAttribute('data-locked-case');
const tick = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));

async function choose(page, index) {
  await page.locator('#case').selectOption(String(index));
  await tick(page);
}
async function point(page, selector, fraction=0) {
  return page.locator(selector).evaluate((node, f) => {
    const a=node.points[0], b=node.points[1] || a, r=node.ownerSVGElement.getBoundingClientRect();
    return {x:r.left+a.x+(b.x-a.x)*f, y:r.top+a.y+(b.y-a.y)*f};
  }, fraction);
}
async function checkLayout(page) {
  const result = await page.evaluate(() => {
    const violations=[];
    if (document.documentElement.scrollWidth>innerWidth+1) violations.push('horizontal overflow');
    for (const svg of document.querySelectorAll('#load-plot, #arch-plot, #coverage')) {
      if (!svg.getBoundingClientRect().height) continue;
      const {width,height}=svg.viewBox.baseVal;
      for (const node of svg.children) {
        const b=node.getBBox();
        if (b.x < -1 || b.y < -1 || b.x+b.width>width+1 || b.y+b.height>height+1) {
          violations.push(svg.id+': '+node.tagName+' '+(node.textContent || '').slice(0,30));
        }
      }
    }
    return violations;
  });
  assert.deepEqual(result, []);
}

(async () => {
  fs.mkdirSync(output, {recursive:true});
  const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL || undefined});
  try {
    const context=await browser.newContext({viewport:{width:1400,height:1000}});
    context.on('page', page => {
      page.on('pageerror', e=>errors.push(e.message));
      page.on('request', r=>{if (/^https?:/.test(r.url())) externalRequests.push(r.url());});
    });
    const page=await context.newPage();
    await page.goto(pathToFileURL(source).href);
    await tick(page);
    assert.equal(await page.locator('#family-tab').getAttribute('aria-selected'), 'true');
    assert.equal(await page.locator('[data-family-case]').count(), data[0].cases.length);
    assert.equal(await page.locator('[data-construction]').count(), 0);
    assert.equal(await page.locator('[data-preview-block]').count(), 0);
    await checkLayout(page);
    await page.screenshot({path:path.join(output,'family-desktop.png'),fullPage:true});

    // Hover from load to line, leave to restore, click to persist.
    const initial=await locked(page);
    let target=data[0].cases[70].id;
    const p=await point(page,'[data-boundary-case="'+target+'"]');
    await page.locator('#load-plot').evaluate(svg=>svg.addEventListener('pointermove',event=>{
      const r=svg.getBoundingClientRect(),x=event.clientX-r.left,y=event.clientY-r.top;
      const points=[...svg.querySelectorAll('[data-boundary-case]')];
      points.sort((a,b)=>(a.points[0].x-x)**2+(a.points[0].y-y)**2-
        ((b.points[0].x-x)**2+(b.points[0].y-y)**2));
      svg.dataset.expectedNearest=points[0].dataset.boundaryCase;
    }));
    await page.mouse.move(p.x,p.y); await tick(page);
    // Browser pointer coordinates may be rounded between densely spaced samples.
    target=await page.locator('#load-plot').getAttribute('data-expected-nearest');
    assert.notEqual(target,initial);
    assert.equal(await active(page),target);
    assert.equal(await locked(page),initial);
    assert.equal(await page.locator('[data-active-path]').getAttribute('data-active-path'),target);
    await page.mouse.move(5,5); await tick(page);
    assert.equal(await active(page),initial);
    await page.mouse.click(p.x,p.y); await tick(page);
    assert.equal(await locked(page),target);

    // Choose an unambiguous path midpoint for the reverse interaction.
    const hit=await page.evaluate(() => {
      const paths=[...document.querySelectorAll('[data-family-case]')];
      const lockedId=document.querySelector('[data-locked-case]').getAttribute('data-locked-case');
      const dist=(p,a,b)=>{
        const dx=b.x-a.x,dy=b.y-a.y,d=dx*dx+dy*dy;
        const t=d?Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/d)):0;
        return (p.x-a.x-t*dx)**2+(p.y-a.y-t*dy)**2;
      };
      for(const node of paths.filter((p,i)=>i%17===0 && p.dataset.familyCase!==lockedId)) {
        for(let j=1;j<node.points.length;j++) {
          const a=node.points[j-1],b=node.points[j],p={x:(a.x+b.x)/2,y:(a.y+b.y)/2};
          const separation=Math.min(...paths.filter(n=>n!==node).map(n=>{
            let d=Infinity;for(let k=1;k<n.points.length;k++) d=Math.min(d,dist(p,n.points[k-1],n.points[k]));
            return d;
          }));
          if(separation>1e-4) {
            const r=node.ownerSVGElement.getBoundingClientRect();
            return {x:p.x+r.left,y:p.y+r.top,id:node.dataset.familyCase};
          }
        }
      }
      return null;
    });
    assert.ok(hit,'a distinct family member must be selectable from the arch');
    await page.mouse.move(hit.x,hit.y); await tick(page);
    assert.equal(await active(page),hit.id);
    assert.equal(await locked(page),target);
    await page.mouse.click(hit.x,hit.y);await tick(page);
    assert.equal(await locked(page),hit.id);
    const pathColor=await page.locator('[data-active-path]').getAttribute('stroke');
    assert.equal(await page.locator('[data-active-case]').getAttribute('fill'),pathColor);

    await page.locator('#selected-line').click();
    assert.equal(await page.locator('[data-family-case]').count(),0);
    assert.equal(await page.locator('[data-active-path]').count(),1);
    const frame=await page.locator('[data-block="0"]').getAttribute('points');
    await page.locator('#construction').check();
    assert.equal(await page.locator('[data-construction]').count(),1);
    assert.equal(await page.locator('[data-concurrency]').count(),19);
    assert.equal(await page.locator('[data-block="0"]').getAttribute('points'),frame);
    await checkLayout(page);

    // Every exact vertex keeps both branches and mode switching keeps the load.
    const vertex=data[0].cases.findIndex(c=>c.kind==='vertex');
    await choose(page,vertex);
    const vertexId=await locked(page);
    await page.locator('#failure-tab').click();
    assert.equal(await active(page),vertexId);
    assert.equal(await page.locator('#branch option').count(),2);
    assert.equal(await page.locator('[data-construction]').count(),0);
    assert.equal(await page.locator('[data-preview-block]').count(),20);
    await page.locator('#branch').selectOption('1');
    const pose=await page.locator('[data-preview-block]').last().getAttribute('points');
    const failureFrame=await page.locator('[data-block="0"]').getAttribute('points');
    await page.locator('#amplitude').fill('20');
    assert.match(await page.locator('#amplitude-label').textContent(),/2.0°/);
    assert.notEqual(await page.locator('[data-preview-block]').last().getAttribute('points'),pose);
    assert.equal(await page.locator('[data-block="0"]').getAttribute('points'),failureFrame);
    await page.locator('#geometry').selectOption('1');
    await choose(page,50);
    const catenaryId=await locked(page);
    await page.locator('#geometry').selectOption('0');
    assert.equal(await locked(page),vertexId);
    assert.equal(await page.locator('#branch').inputValue(),'1');
    assert.equal(await page.locator('#amplitude').inputValue(),'20');
    await page.locator('#family-tab').click();
    assert.equal(await page.locator('[data-preview-block]').count(),0);
    assert.equal(await locked(page),vertexId);
    assert.equal(await page.locator('[data-block="0"]').getAttribute('points'),frame);
    await page.locator('#failure-tab').click();
    assert.equal(await page.locator('#branch').inputValue(),'1');
    await page.locator('#amplitude').fill('100');
    await checkLayout(page);
    await page.screenshot({path:path.join(output,'failure-desktop.png'),fullPage:true});

    // All selectable cases and both branches fit the fixed failure frame.
    for (let gi=0;gi<data.length;gi++) {
      await page.locator('#geometry').selectOption(String(gi));
      if(gi===1) assert.equal(await locked(page),catenaryId);
      const before=await page.locator('[data-block="0"]').getAttribute('points');
      for (let i=0;i<data[gi].cases.length;i++) {
        await page.locator('#case').selectOption(String(i));
        if(data[gi].cases[i].kind==='vertex') {
          assert.equal(await page.locator('#branch option').count(),2);
          await page.locator('#branch').selectOption('1');
        }
        assert.equal(await page.locator('[data-block="0"]').getAttribute('points'),before);
        await checkLayout(page);
      }
    }

    // Keyboard controls and all display variants across themes and screen sizes.
    await page.locator('#family-tab').click();
    await page.locator('#whole-family').click();
    await page.locator('#family-tab').focus(); await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#failure-tab').getAttribute('aria-selected'),'true');
    await page.keyboard.press('Home');
    assert.equal(await page.locator('#family-tab').getAttribute('aria-selected'),'true');
    const oldIndex=Number(await page.locator('#load-slider').inputValue());
    await page.locator('#load-slider').focus();await page.keyboard.press('ArrowLeft');
    assert.equal(Number(await page.locator('#case').inputValue()),oldIndex-1);
    for(const width of [1400,768,360,320]) for(const theme of ['light','dark']) {
      await page.setViewportSize({width,height:1000});await page.emulateMedia({colorScheme:theme});await tick(page);
      for(const tab of ['family','failure']) {
        await page.locator('#'+tab+'-tab').click();await tick(page);await checkLayout(page);
      }
    }
    await page.locator('#family-tab').click();
    await page.screenshot({path:path.join(output,'family-mobile-dark.png'),fullPage:true});

    // Missing opening modes still have a working family and disabled preview.
    const empty=structuredClone(data);
    empty[0].modes=[];empty[0].cases.forEach(c=>{c.modes=[];c.status='no verified opening mode; contact limits only';});
    const unavailable=await context.newPage();
    await unavailable.setContent(html.replace(jsonPattern,(_,a,b,c)=>a+JSON.stringify(empty)+c));await tick(unavailable);
    assert.equal(await unavailable.locator('[data-family-case]').count(),empty[0].cases.length);
    await unavailable.locator('#failure-tab').click();
    assert.equal(await unavailable.locator('#amplitude').isDisabled(),true);
    assert.equal(await unavailable.locator('[data-preview-block]').count(),0);
    assert.match(await unavailable.locator('#status').textContent(),/no verified opening mode/);

    // Exact coincident paths retain the current selection during hover and click.
    const overlap=structuredClone(data);
    overlap[0].cases[1].pressure=overlap[0].cases[0].pressure;
    const tied=await context.newPage();
    await tied.setContent(html.replace(jsonPattern,(_,a,b,c)=>a+JSON.stringify(overlap)+c));
    await choose(tied,1);
    const tiedPoint=await point(tied,'[data-active-path]',0.5);
    await tied.mouse.move(tiedPoint.x,tiedPoint.y);await tick(tied);
    assert.equal(await active(tied),overlap[0].cases[1].id);
    await tied.mouse.click(tiedPoint.x,tiedPoint.y);await tick(tied);
    assert.equal(await locked(tied),overlap[0].cases[1].id);

    const touch=await browser.newContext({viewport:{width:390,height:900},hasTouch:true,isMobile:true});
    const mobile=await touch.newPage();mobile.on('pageerror',e=>errors.push(e.message));
    await mobile.goto(pathToFileURL(source).href);await tick(mobile);
    const tp=await point(mobile,'[data-boundary-case="'+data[0].cases[90].id+'"]');
    await mobile.locator('#load-plot').evaluate(svg=>svg.addEventListener('click',event=>{
      const r=svg.getBoundingClientRect(),x=event.clientX-r.left,y=event.clientY-r.top;
      const points=[...svg.querySelectorAll('[data-boundary-case]')];
      points.sort((a,b)=>(a.points[0].x-x)**2+(a.points[0].y-y)**2-
        ((b.points[0].x-x)**2+(b.points[0].y-y)**2));
      svg.dataset.expectedNearest=points[0].dataset.boundaryCase;
    }));
    await mobile.touchscreen.tap(tp.x,tp.y);await tick(mobile);
    assert.equal(await locked(mobile),await mobile.locator('#load-plot').getAttribute('data-expected-nearest'));
    assert.notEqual(await locked(mobile),data[0].cases[0].id);
    assert.equal(await active(mobile),await locked(mobile));
    await checkLayout(mobile);
    assert.deepEqual(errors,[]);assert.deepEqual(externalRequests,[]);
    console.log('PASS: linked selection, all boundary cases, branches, unavailable/overlap cases, keyboard/touch, themes/layouts, offline.');
    console.log('Screenshots: '+output);
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});

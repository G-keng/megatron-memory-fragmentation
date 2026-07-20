"""Generate a self-contained interactive allocator dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_dashboard(data: Dict[str, Any], path: Path) -> None:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    path.write_text(_HTML.replace("__MEMORY_DATA__", payload), encoding="utf-8")


_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU Memory Fragmentation Dashboard</title>
<style>
:root{--bg:#f3f6fb;--panel:#fff;--ink:#172033;--muted:#667085;--line:#d8e0ed;--blue:#3974d6;--red:#dc4c64;--orange:#e89a3c;--gray:#b8c0cc;--navy:#213b64}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:18px 24px;background:var(--navy);color:#fff;display:flex;align-items:center;justify-content:space-between;gap:16px}h1{font-size:20px;margin:0}.badge{padding:5px 10px;border-radius:99px;font-weight:700;background:#dff7e8;color:#167044}.badge.approx{background:#fff0d7;color:#945b00}
.controls{display:flex;gap:16px;flex-wrap:wrap;padding:14px 24px;background:#fff;border-bottom:1px solid var(--line)}label{display:flex;align-items:center;gap:7px;color:var(--muted)}input,select{border:1px solid var(--line);border-radius:7px;padding:6px 8px;background:#fff;color:var(--ink)}
.warning{margin:14px 24px 0;padding:10px 13px;border-radius:8px;background:#fff4dc;color:#7b5100;border:1px solid #f0d397}.hidden{display:none!important}
main{padding:14px 24px 28px;display:grid;grid-template-columns:310px minmax(0,1fr);gap:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 8px #20304a0d}.panel h2{font-size:14px;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}
.timeline-panel{grid-column:1/-1;padding-bottom:8px}.timeline-wrap{padding:10px 14px 2px}#timeline{width:100%;height:220px;display:block}.axis{stroke:#aeb8c7;stroke-width:1}.timeline-line{fill:none;stroke:var(--blue);stroke-width:2}.timeline-lower{fill:none;stroke:var(--orange);stroke-width:1.4;stroke-dasharray:5 4}.threshold-line{stroke:var(--red);stroke-width:1.5;stroke-dasharray:7 5}.timeline-point{fill:#fff;stroke:var(--red);stroke-width:2;cursor:pointer}
#moment-list{padding:8px;max-height:690px;overflow:auto}.moment{width:100%;text-align:left;border:1px solid transparent;border-radius:8px;padding:10px;margin-bottom:7px;background:#f7f9fc;color:var(--ink);cursor:pointer}.moment:hover{border-color:#a9bfe6}.moment.selected{border-color:var(--blue);background:#edf4ff}.moment-title{display:flex;justify-content:space-between;font-weight:700}.moment-meta{margin-top:5px;color:var(--muted);font-size:12px}.empty{padding:20px;color:var(--muted);text-align:center}
.right{display:grid;gap:14px;min-width:0}.metrics{display:grid;grid-template-columns:repeat(6,minmax(105px,1fr));gap:8px;padding:12px}.metric{background:#f7f9fc;border-radius:8px;padding:9px}.metric strong{display:block;font-size:16px;margin-top:3px}.metric span{font-size:11px;color:var(--muted)}
.legend{display:flex;gap:15px;flex-wrap:wrap;padding:9px 14px;border-bottom:1px solid var(--line);color:var(--muted);font-size:12px}.swatch{width:11px;height:11px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:-1px}.active{background:var(--blue)}.pending{background:var(--orange)}.stranded{background:var(--red)}.cache{background:var(--gray)}
#segments{padding:10px 12px;max-height:480px;overflow:auto}.segment{display:grid;grid-template-columns:205px minmax(300px,1fr);gap:9px;align-items:center;margin:5px 0}.segment-label{font:11px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.segment-bar{height:27px;display:flex;border:1px solid #8e9bad;border-radius:4px;overflow:hidden;background:#e8edf4}.block{height:100%;border:0;border-right:1px solid #ffffff7d;min-width:1px;padding:0;cursor:pointer}.block.pinner{outline:2px solid #7c3aed;outline-offset:-2px}.block:focus{outline:2px solid #111827;outline-offset:-2px}
#details{padding:12px 14px;min-height:120px;white-space:pre-wrap;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;color:#31405b;overflow:auto}.phase{font-weight:700;color:var(--blue)}
@media(max-width:900px){main{grid-template-columns:1fr}.timeline-panel{grid-column:1}.metrics{grid-template-columns:repeat(2,1fr)}.segment{grid-template-columns:1fr}.segment-label{white-space:normal}}
</style>
</head>
<body>
<header><div><h1>GPU Memory Fragmentation Dashboard</h1><div id="subtitle"></div></div><span id="mode-badge" class="badge"></span></header>
<div class="controls">
  <label>Top-K <input id="top-k" data-testid="top-k" type="number" min="1" max="100" value="10"></label>
  <label>最小碎片率 <input id="threshold" data-testid="threshold" type="number" min="0" max="100" step="0.1" value="0">%</label>
  <label>逻辑阶段 <select id="phase-filter" data-testid="phase-filter"><option value="all">全部阶段</option></select></label>
</div>
<div id="warning" class="warning hidden"></div>
<main>
  <section class="panel timeline-panel"><h2>碎片率时间线</h2><div class="timeline-wrap"><svg id="timeline" data-testid="timeline" viewBox="0 0 1000 220" role="img" aria-label="碎片率时间线"></svg></div></section>
  <section class="panel"><h2>Top-K 时刻</h2><div id="moment-list" data-testid="moment-list"></div></section>
  <div class="right">
    <section class="panel"><h2 id="moment-heading">时刻详情</h2><div id="metrics" class="metrics"></div></section>
    <section class="panel"><h2>Allocator 布局</h2><div class="legend"><span><i class="swatch active"></i>Active</span><span><i class="swatch pending"></i>Pending free</span><span><i class="swatch stranded"></i>Stranded</span><span><i class="swatch cache"></i>Releasable cache</span><span>紫色边框：候选 pinning allocation</span></div><div id="segments" data-testid="segments"></div></section>
    <section class="panel"><h2>Block / 阶段信息</h2><div id="details" data-testid="details">点击一个 block 查看详细信息。</div></section>
  </div>
</main>
<script>
const DATA=__MEMORY_DATA__;
const el=id=>document.getElementById(id);
let selected=null;
const fmtBytes=n=>{let x=Number(n||0),u=['B','KiB','MiB','GiB','TiB'],i=0;while(Math.abs(x)>=1024&&i<u.length-1){x/=1024;i++}return `${x.toFixed(i?2:0)} ${u[i]}`};
const fmtPct=x=>`${(Number(x||0)*100).toFixed(2)}%`;
const fmtRange=m=>{const hi=Number(m.fragmentation_ratio||0),lo=Number(m.fragmentation_ratio_lower??hi);return Math.abs(hi-lo)<1e-12?fmtPct(hi):`${fmtPct(lo)}–${fmtPct(hi)}`};
const phaseMeta=m=>Object.entries(m||{}).map(([k,v])=>`${k}=${v}`).join(' · ');

function init(){
  el('top-k').value=DATA.top_k||10;
  el('threshold').value=((DATA.initial_threshold||0)*100).toFixed(1);
  el('mode-badge').textContent=DATA.exact?'EXACT':'APPROXIMATE';
  if(!DATA.exact){
    el('mode-badge').classList.add('approx');el('warning').classList.remove('hidden');
    el('warning').textContent=DATA.history_mode==='reverse_approximate'
      ? `精确正向重放失败：${DATA.replay_error||'历史不完整'}。当前使用默认配置感知的反向重放；蓝线和 Top-K 使用碎片率保守上界，橙色虚线为估计下界。`
      : `当前 trace 无法进行历史重放：${DATA.reverse_error||DATA.replay_error||'历史不可用'}。只显示最终 snapshot。`;
  }
  el('subtitle').textContent=`${DATA.moments.length} 个可查看时刻 · 默认 Top-${DATA.top_k}`;
  const phases=[...new Set(DATA.moments.map(x=>x.primary_phase))].sort();
  phases.forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent=p;el('phase-filter').appendChild(o)});
  el('top-k').addEventListener('input',renderList);
  el('threshold').addEventListener('input',()=>{renderList();renderTimeline()});
  el('phase-filter').addEventListener('change',renderList);
  selected=DATA.moments[0]||null;
  renderList();renderTimeline();renderMoment();
}

function filteredMoments(){
  const k=Math.max(1,Number(el('top-k').value)||10),threshold=(Number(el('threshold').value)||0)/100,phase=el('phase-filter').value;
  return DATA.moments.filter(m=>m.fragmentation_ratio>=threshold&&(phase==='all'||m.primary_phase===phase)).slice(0,k);
}

function renderList(){
  const root=el('moment-list'),items=filteredMoments();root.replaceChildren();
  if(!items.length){const d=document.createElement('div');d.className='empty';d.textContent='当前阈值或阶段下没有候选时刻。';root.appendChild(d);return}
  if(!selected||!items.includes(selected)){selected=items[0];renderMoment()}
  items.forEach(m=>{const b=document.createElement('button');b.className='moment'+(m===selected?' selected':'');b.dataset.testid=`moment-${m.rank}`;b.innerHTML=`<div class="moment-title"><span>#${m.rank} · ${fmtRange(m)}</span><span>${fmtBytes(m.stranded_free_bytes)}</span></div><div class="moment-meta">${m.primary_phase}<br>event ${m.event_index} · device ${m.device}${phaseMeta(m.phase_metadata)?' · '+phaseMeta(m.phase_metadata):''}</div>`;b.addEventListener('click',()=>{selected=m;renderList();renderMoment();renderTimeline()});root.appendChild(b)})
}

function renderTimeline(){
  const svg=el('timeline'),rows=DATA.timeline;svg.replaceChildren();
  if(!rows.length){svg.innerHTML='<text x="500" y="110" text-anchor="middle" fill="#667085">无可用时间线</text>';return}
  const pad={l:58,r:18,t:15,b:30},w=1000-pad.l-pad.r,h=220-pad.t-pad.b,minT=Math.min(...rows.map(x=>x.time_us)),maxT=Math.max(...rows.map(x=>x.time_us)),maxY=Math.max(.01,...rows.map(x=>x.fragmentation_ratio),...(DATA.moments.map(x=>x.fragmentation_ratio)))*1.08;
  const X=t=>pad.l+(maxT===minT ? 0.5 : (t-minT)/(maxT-minT))*w,Y=y=>pad.t+h-(y/maxY)*h,ns='http://www.w3.org/2000/svg';
  const line=(x1,y1,x2,y2,cls)=>{const n=document.createElementNS(ns,'line');Object.entries({x1,y1,x2,y2,class:cls}).forEach(([k,v])=>n.setAttribute(k,v));svg.appendChild(n)};
  line(pad.l,pad.t+h,pad.l+w,pad.t+h,'axis');line(pad.l,pad.t,pad.l,pad.t+h,'axis');
  const threshold=(Number(el('threshold').value)||0)/100;line(pad.l,Y(threshold),pad.l+w,Y(threshold),'threshold-line');
  const byDevice=new Map();rows.forEach(r=>{if(!byDevice.has(r.device))byDevice.set(r.device,[]);byDevice.get(r.device).push(r)});
  byDevice.forEach(values=>{const p=document.createElementNS(ns,'polyline');p.setAttribute('class','timeline-line');p.setAttribute('points',values.map(r=>`${X(r.time_us)},${Y(r.fragmentation_ratio)}`).join(' '));svg.appendChild(p);if(DATA.history_mode==='reverse_approximate'){const low=document.createElementNS(ns,'polyline');low.setAttribute('class','timeline-lower');low.setAttribute('points',values.map(r=>`${X(r.time_us)},${Y(r.fragmentation_ratio_lower??r.fragmentation_ratio)}`).join(' '));svg.appendChild(low)}});
  DATA.moments.forEach(m=>{const c=document.createElementNS(ns,'circle');c.setAttribute('class','timeline-point');c.setAttribute('cx',X(m.time_us));c.setAttribute('cy',Y(m.fragmentation_ratio));c.setAttribute('r',m===selected?7:5);c.dataset.testid=`timeline-moment-${m.rank}`;c.addEventListener('click',()=>{selected=m;renderList();renderMoment();renderTimeline()});svg.appendChild(c)});
  const labels=[[0,'0%'],[maxY/2,fmtPct(maxY/2)],[maxY,fmtPct(maxY)]];labels.forEach(([v,t])=>{const n=document.createElementNS(ns,'text');n.setAttribute('x',pad.l-8);n.setAttribute('y',Y(v)+4);n.setAttribute('text-anchor','end');n.setAttribute('fill','#667085');n.setAttribute('font-size','11');n.textContent=t;svg.appendChild(n)});
}

function metric(label,value){const d=document.createElement('div');d.className='metric';d.innerHTML=`<span>${label}</span><strong>${value}</strong>`;return d}
function renderMoment(){
  const metrics=el('metrics'),segments=el('segments');metrics.replaceChildren();segments.replaceChildren();
  if(!selected){el('moment-heading').textContent='时刻详情';return}
  el('moment-heading').textContent=`#${selected.rank} · ${selected.primary_phase}`;
  metrics.append(metric('碎片率上界',fmtPct(selected.fragmentation_ratio)),metric('估计区间',fmtRange(selected)),metric('Stranded 上界',fmtBytes(selected.stranded_free_bytes)),metric('Reserved',fmtBytes(selected.reserved_bytes)),metric('Releasable cache',fmtBytes(selected.releasable_cache_bytes)),metric('Pending free',fmtBytes(selected.pending_free_bytes)));
  const ordered=[...(selected.segments||[])].sort((a,b)=>Number(b.partially_active)-Number(a.partially_active)||b.total_size-a.total_size);
  ordered.forEach(seg=>{
    const row=document.createElement('div');row.className='segment';
    const label=document.createElement('div');label.className='segment-label';label.title=seg.address;label.textContent=`${seg.address} · ${fmtBytes(seg.total_size)} · s${seg.stream}`;
    const bar=document.createElement('div');bar.className='segment-bar';
    const activeBlocks=seg.blocks.filter(b=>b.state!=='inactive'),smallest=activeBlocks.reduce((best,b)=>!best||b.size<best.size?b:best,null);
    seg.blocks.forEach(block=>{const btn=document.createElement('button');const pending=block.state.includes('pending')||block.state.includes('awaiting');const cls=block.state==='inactive'?(seg.partially_active?'stranded':'cache'):(pending?'pending':'active');btn.className=`block ${cls}`+(seg.partially_active&&block===smallest?' pinner':'');btn.style.width=`${Math.max(.08,block.size/seg.total_size*100)}%`;const uncertainty=block.uncertainty_bytes?` · uncertainty ≤ ${fmtBytes(block.uncertainty_bytes)}`:'';btn.title=`${block.address} · ${fmtBytes(block.size)} · ${block.state}${uncertainty}`;btn.setAttribute('aria-label',btn.title);btn.addEventListener('click',()=>showBlock(seg,block));bar.appendChild(btn)});
    row.append(label,bar);segments.appendChild(row)
  });
  el('details').textContent=`history mode: ${selected.history_mode||DATA.history_mode}\nphase: ${selected.primary_phase}\nconfidence: ${selected.phase_confidence}\n${phaseMeta(selected.phase_metadata)}\nevent: ${selected.event_index}\ntime_us: ${selected.time_us}\ndevice: ${selected.device}\nfragmentation interval: ${fmtRange(selected)}\nuncertainty: ${fmtBytes(selected.fragmentation_uncertainty_bytes||0)}`;
}
function showBlock(seg,block){el('details').textContent=`segment: ${seg.address}\nsegment size: ${fmtBytes(seg.total_size)}\nstream: ${seg.stream}\nblock: ${block.address}\nblock size: ${fmtBytes(block.size)}\nrequested: ${fmtBytes(block.requested_size)}\nstate: ${block.state}\nreconstructed: ${Boolean(block.reconstructed)}\nuncertainty: ${fmtBytes(block.uncertainty_bytes||0)}\n\n${(block.frames||[]).join('\n')||'no allocation frames'}`}
init();
</script>
</body>
</html>"""

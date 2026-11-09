from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()
app = FastAPI(title="Daily Gusle", docs_url=None, redoc_url=None)


class EditionRequest(BaseModel):
    topics: list[str] = Field(min_length=1, max_length=8)
    language: str = "Thai"
    tavily_api_key: str | None = None
    openai_api_key: str | None = None
    model: str | None = None


def clean_text(value: Any, limit: int = 1000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def fetch_sources(topic: str, api_key: str) -> list[dict[str, Any]]:
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": topic,
            "topic": "news",
            "search_depth": "advanced",
            "days": 14,
            "max_results": 7,
            "include_images": True,
            "include_answer": False,
        },
        timeout=45,
    )
    response.raise_for_status()
    found = []
    for result in response.json().get("results", []):
        url = clean_text(result.get("url"), 2000)
        if not url.startswith(("https://", "http://")):
            continue
        host = urlparse(url).netloc.removeprefix("www.")
        found.append({
            "title": clean_text(result.get("title"), 300),
            "url": url,
            "source": host,
            "date": clean_text(result.get("published_date") or result.get("date"), 80),
            "content": clean_text(result.get("content"), 2200),
            "image": clean_text(result.get("image"), 2000),
        })
    return found


def write_stories(topic: str, sources: list[dict[str, Any]], api_key: str, model: str) -> list[dict[str, Any]]:
    evidence = [
        {"id": i, "title": s["title"], "publisher": s["source"], "published_date": s["date"] or "not provided",
         "url": s["url"], "article_excerpt": s["content"]}
        for i, s in enumerate(sources, 1)
    ]
    system = (
        "You are a careful newspaper reporter. Write in the requested language, using ONLY the supplied source excerpts. "
        "Never invent facts, dates, quotes, or source details. If a date is absent, do not infer one. "
        "Return JSON only with {stories:[{headline,dek,paragraphs:[...],source_ids:[integer,...]}]}. "
        "Create one concise, useful newspaper story for this assignment. Include [n] citations in the relevant paragraphs, "
        "where n is a supplied source ID. Cite every material factual claim, use at least one source, and do not add an editor's note."
    )
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"language": topic.rsplit("\nLANGUAGE: ", 1)[-1], "assignment": topic.split("\nLANGUAGE: ", 1)[0], "sources": evidence}, ensure_ascii=False)},
            ],
        },
        timeout=90,
    )
    response.raise_for_status()
    data = json.loads(response.json()["choices"][0]["message"]["content"])
    stories = data.get("stories", [])
    if isinstance(stories, dict):
        stories = [stories]
    valid = []
    for story in stories:
        if not isinstance(story, dict):
            continue
        ids = []
        for item in story.get("source_ids", []):
            try:
                idx = int(item)
                if 1 <= idx <= len(sources) and idx not in ids:
                    ids.append(idx)
            except (TypeError, ValueError):
                pass
        if not ids:
            ids = [1]
        paragraphs = story.get("paragraphs", [])
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        valid.append({
            "headline": clean_text(story.get("headline"), 300) or "ข่าวเด่น",
            "dek": clean_text(story.get("dek"), 500),
            "paragraphs": [clean_text(p, 2000) for p in paragraphs if clean_text(p, 2000)],
            "source_ids": ids,
        })
    return valid


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/newspaper")
def newspaper(request: EditionRequest) -> dict[str, Any]:
    tavily_key = request.tavily_api_key or os.getenv("TAVILY_API_KEY")
    openai_key = request.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not tavily_key or not openai_key:
        raise HTTPException(400, "กรุณาใส่ Tavily API key และ OpenAI API key หรือกำหนดไว้ในไฟล์ .env")
    all_sources: list[dict[str, Any]] = []
    stories: list[dict[str, Any]] = []
    model = request.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        for topic in request.topics:
            sources = fetch_sources(topic, tavily_key)
            if not sources:
                continue
            drafted = write_stories(f"{topic}\nLANGUAGE: {request.language}", sources, openai_key, model)
            offset = len(all_sources)
            for story in drafted:
                story["source_ids"] = [i + offset for i in story["source_ids"]]
                stories.append(story)
            all_sources.extend(sources)
    except requests.HTTPError as exc:
        detail = "ผู้ให้บริการปฏิเสธคำขอ ตรวจสอบ API key และโควตาอีกครั้ง"
        if exc.response is not None and exc.response.text:
            detail += f" ({exc.response.status_code})"
        raise HTTPException(502, detail) from exc
    except requests.RequestException as exc:
        raise HTTPException(502, "เชื่อมต่อบริการค้นหาข่าวหรือ AI ไม่สำเร็จ ลองอีกครั้ง") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(502, "ไม่สามารถอ่านผลจากบริการ AI ได้ ลองอีกครั้ง") from exc
    if not stories:
        raise HTTPException(404, "ไม่พบข่าวที่ใช้ได้ในช่วงค้นหานี้ ลองหัวข้ออื่น")
    return {"date": date.today().isoformat(), "stories": stories, "sources": all_sources}


PAGE = r'''<!doctype html>
<html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Daily Gusle — หนังสือพิมพ์ประจำวัน</title>
<style>
:root{--ink:#171713;--muted:#625e55;--rule:#28251f;--paper:#fffdf7;--accent:#8d2b20}*{box-sizing:border-box}body{margin:0;background:#e8e5dc;color:var(--ink);font-family:Georgia,"Times New Roman",serif}.controls{max-width:1100px;margin:22px auto;padding:20px;background:#fff;border:1px solid #ccc;font:15px system-ui,sans-serif}.controls h1{margin:0 0 6px;font:700 25px system-ui,sans-serif}.controls p{margin:0 0 15px;color:#555}.fields{display:grid;grid-template-columns:1fr 1fr;gap:10px}.fields label{font-size:13px;font-weight:650}.fields input,.fields textarea{display:block;width:100%;margin-top:5px;padding:10px;border:1px solid #aaa;border-radius:4px;font:14px system-ui,sans-serif}.fields textarea{height:74px;resize:vertical}.actions{display:flex;gap:10px;margin-top:12px;align-items:center}.actions button{padding:10px 16px;border:0;border-radius:4px;background:#222;color:#fff;font-weight:700;cursor:pointer}.actions button.secondary{background:#eee;color:#222;border:1px solid #aaa}.actions span{font:13px system-ui,sans-serif;color:#555}#paper{max-width:1100px;min-height:600px;margin:0 auto 30px;padding:34px 40px;background:var(--paper);box-shadow:0 2px 15px #0002}.masthead{text-align:center;border-bottom:4px double var(--rule);padding-bottom:10px}.brand{margin:0;font-size:clamp(44px,8vw,82px);font-weight:900;letter-spacing:-4px;line-height:.95}.tagline{margin:7px 0 0;font:italic 14px Georgia,serif}.editionline{display:flex;justify-content:space-between;margin-top:14px;padding:7px 0;border-top:1px solid var(--rule);font:bold 11px Arial,sans-serif;text-transform:uppercase;letter-spacing:.07em}.edition-title{text-align:center;margin:15px 0 20px;font:bold 12px Arial,sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}.stories{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 20px;align-items:start}.story{min-width:0;padding:0 0 13px;margin:0 0 14px;border-bottom:1px solid #aaa;break-inside:avoid;page-break-inside:avoid}.story.lead{grid-column:span 2}.story h2{font-size:26px;line-height:1.04;margin:4px 0 7px;letter-spacing:-.4px}.story:not(.lead) h2{font-size:20px}.dek{margin:0 0 8px;font:bold 14px/1.35 Georgia,serif}.byline{font:10px Arial,sans-serif;color:var(--muted);margin:7px 0;text-transform:uppercase;letter-spacing:.04em}.story p{font-size:13px;line-height:1.48;margin:7px 0;text-align:justify}.story img{display:block;width:100%;max-height:300px;object-fit:cover;margin:7px 0 3px;filter:grayscale(100%)}.caption{font:italic 10px Arial,sans-serif!important;color:var(--muted);margin:3px 0!important}.sources{border-top:1px dotted #999;padding-top:5px;margin-top:8px;font:9px/1.4 Arial,sans-serif;color:var(--muted)}.sources a,.story a{color:inherit}.sources strong{color:var(--ink)}.empty{text-align:center;color:#777;padding:80px 10px;font-size:20px}.footer{border-top:3px double var(--rule);padding-top:7px;text-align:center;font:9px Arial,sans-serif;color:var(--muted)}
@media(max-width:700px){#paper{padding:22px 18px;margin:0 8px 20px}.stories{grid-template-columns:repeat(2,minmax(0,1fr));gap:0 14px}.story.lead{grid-column:span 2}.fields{grid-template-columns:1fr}.brand{letter-spacing:-2px}.story h2{font-size:22px}}
@page{size:A4 portrait;margin:11mm 10mm 13mm}@media print{body{background:#fff;color:#000;-webkit-print-color-adjust:exact;print-color-adjust:exact}.controls{display:none!important}#paper{max-width:none;min-height:0;margin:0;padding:0;box-shadow:none;background:#fffdf7}.stories{grid-template-columns:repeat(3,minmax(0,1fr));gap:0 4mm}.story.lead{grid-column:span 2}.story{padding-bottom:2mm;margin-bottom:2mm}.story h2{font-size:17pt}.story:not(.lead) h2{font-size:13pt}.story p{font-size:9pt;line-height:1.35}.dek{font-size:10pt}.story img{max-height:58mm}.masthead{padding-bottom:2mm}.brand{font-size:48pt}.edition-title{margin:3mm 0}.footer{margin-top:2mm}a{text-decoration:none}}
</style></head><body>
<section class="controls"><h1>Daily Gusle — ห้องข่าวประจำวัน</h1><p>ค้นข่าวล่าสุด เขียนพร้อมอ้างอิง และจัดหน้าแน่นแบบหนังสือพิมพ์ พร้อมพิมพ์หรือบันทึกเป็น PDF</p>
<div class="fields"><label>หัวข้อข่าว (แยกแต่ละหัวข้อด้วยบรรทัดใหม่)<textarea id="topics" placeholder="เทคโนโลยี AI&#10;ข่าวโลก&#10;เศรษฐกิจ"></textarea></label><label>ภาษา<input id="language" value="Thai"></label>
<label>Tavily API key<input id="tavily" type="password" placeholder="เว้นว่างหากตั้งค่าใน .env"></label><label>OpenAI API key<input id="openai" type="password" placeholder="เว้นว่างหากตั้งค่าใน .env"></label></div>
<div class="actions"><button id="generate">จัดทำฉบับวันนี้</button><button class="secondary" id="print">พิมพ์ / บันทึก PDF</button><span id="status">ใส่หัวข้อข่าวอย่างน้อยหนึ่งหัวข้อ</span></div></section>
<main id="paper"><header class="masthead"><h1 class="brand">Daily Gusle</h1><p class="tagline">ข่าววันนี้ อ่านอย่างมีที่มา</p><div class="editionline"><span>ฉบับประจำวัน</span><span id="date">—</span><span>ราคา: อ่านฟรี</span></div></header><div class="edition-title">ข่าวเด่น · รายงานจากแหล่งข่าว</div><section id="stories" class="stories"><div class="empty">เริ่มจัดทำฉบับข่าวของคุณได้จากแบบฟอร์มด้านบน</div></section><footer class="footer">ข่าวทุกชิ้นมีลิงก์อ้างอิงและวันที่เผยแพร่จากแหล่งข่าว · ตรวจสอบต้นฉบับก่อนนำไปใช้อ้างอิง</footer></main>
<script>
const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function cited(text,sources){return esc(text).replace(/\[(\d+)\]/g,(m,n)=>{let i=Number(n)-1,s=sources[i];return s?`<a href="${esc(s.url)}" target="_blank" rel="noopener">[${n}]</a>`:m})}
function render(data){$('date').textContent=new Date(data.date+'T00:00:00').toLocaleDateString('th-TH',{year:'numeric',month:'long',day:'numeric'});$('stories').innerHTML=data.stories.map((a,i)=>{let ids=[...new Set(a.source_ids||[])].filter(n=>data.sources[n]);let primary=ids.map(n=>data.sources[n]).find(s=>s.date);let image=ids.map(n=>data.sources[n]).find(s=>s.image)?.image;let sourceList=ids.map(n=>{let s=data.sources[n];return `<a href="${esc(s.url)}" target="_blank" rel="noopener">[${n+1}] ${esc(s.source)}${s.date?' · '+esc(s.date):' · ไม่ระบุวันที่'}</a>`}).join(' · ');return `<article class="story ${i===0?'lead':''}"><div class="byline">${esc(primary?.source||'รายงานข่าว')} · เผยแพร่ ${esc(primary?.date||'ไม่ระบุวันที่')}</div><h2>${esc(a.headline)}</h2>${a.dek?`<div class="dek">${esc(a.dek)}</div>`:''}${image?`<img src="${esc(image)}" alt="ภาพประกอบจากแหล่งข่าว" loading="lazy"><p class="caption">ภาพประกอบจากแหล่งข่าว</p>`:''}${(a.paragraphs||[]).map(p=>`<p>${cited(p,data.sources)}</p>`).join('')}<div class="sources"><strong>แหล่งข่าว:</strong> ${sourceList||'ไม่มีข้อมูลแหล่งข่าว'}</div></article>`}).join('')}
$('generate').onclick=async()=>{let topics=$('topics').value.split(/\n+/).map(x=>x.trim()).filter(Boolean);if(!topics.length){$('status').textContent='กรุณาใส่หัวข้อข่าว';return}const btn=$('generate');btn.disabled=true;$('status').textContent='กำลังค้นคว้าและจัดหน้า…';try{let r=await fetch('/api/newspaper',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topics,language:$('language').value||'Thai',tavily_api_key:$('tavily').value||null,openai_api_key:$('openai').value||null})});let d=await r.json();if(!r.ok)throw Error(d.detail||'เกิดข้อผิดพลาด');render(d);$('status').textContent=`จัดทำ ${d.stories.length} ข่าว พร้อมพิมพ์ได้`;window.scrollTo({top:$('paper').offsetTop-10,behavior:'smooth'})}catch(e){$('status').textContent=e.message}finally{btn.disabled=false}};
$('print').onclick=()=>window.print();
</script></body></html>'''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), reload=False)

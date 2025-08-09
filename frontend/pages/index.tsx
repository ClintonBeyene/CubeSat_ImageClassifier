import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import styles from '../styles/Home.module.css';
import Starfield from '../components/Starfield';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/predict';

interface PredictionRecord {
  id: string;
  fileName: string;
  objectUrl: string;
  predicted_class?: string;
  probability?: number;
  probsVector?: number[];
  createdAt: number;
  status: 'pending' | 'done' | 'error';
  error?: string;
  latencyMs?: number;
  // New: client-side metrics
  blurVar?: number;
  noiseLvl?: number;
}

const CLASS_COLORS: Record<string,string> = {
  Blurry: '#f59e0b',
  Corrupt: '#ef4444',
  Missing_Data: '#6366f1',
  Noisy: '#10b981',
  Priority: '#3b82f6'
};

const CLASS_DESCRIPTIONS: Record<string,string> = {
  Blurry: 'Motion/defocus loss of detail.',
  Corrupt: 'Sensor or transmission artifacts.',
  Missing_Data: 'Partial or missing frame regions.',
  Noisy: 'Radiation/electronic noise dominated.',
  Priority: 'Clear, science-ready imagery.'
};

const CLASS_ICONS: Record<string,string> = {
  Blurry: '⚠️',
  Corrupt: '⛔',
  Missing_Data: '⚠️',
  Noisy: '⚠️',
  Priority: '✔️'
};

export default function Home() {
  const [items, setItems] = useState<PredictionRecord[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [theme, setTheme] = useState<'dark'|'light'>(typeof window!=='undefined' && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light':'dark');
  const [compare, setCompare] = useState<string[]>([]);
  const [activeFilters, setActiveFilters] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement|null>(null);

  // Load history
  useEffect(()=>{
    try {
      const raw = localStorage.getItem('cubesat-history');
      if(raw){ setItems(JSON.parse(raw)); }
    } catch{}
  },[]);
  useEffect(()=>{ localStorage.setItem('cubesat-history', JSON.stringify(items.slice(0,100))); },[items]);

  const handleFiles = (fileList: FileList) => {
  const arr = Array.from(fileList);
    arr.forEach(file => queueUpload(file));
  };

  const queueUpload = (file: File) => {
    const id = crypto.randomUUID();
    const objectUrl = URL.createObjectURL(file);
    const record: PredictionRecord = { id, fileName: file.name, objectUrl, createdAt: Date.now(), status: 'pending' };
    setItems(prev=>[record,...prev]);
    // Compute scientific metrics client-side
    computeImageMetrics(objectUrl).then(m => {
      setItems(prev=> prev.map(it => it.id===id ? { ...it, blurVar: m.blurVar, noiseLvl: m.noiseLvl } : it));
    }).catch(()=>{});
    uploadFile(id,file);
  };

  const uploadFile = async (id:string,file:File) => {
    const formData = new FormData(); formData.append('file', file);
    const start = performance.now();
    try {
      const res = await axios.post(API_URL, formData, { headers: { 'Content-Type': 'multipart/form-data' }});
      const latency = performance.now()-start;
    // Use model-provided probability (support 0..1 or 0..100)
    let prob: number | undefined = undefined;
    const raw = (res.data?.probability ?? res.data?.prob ?? undefined);
      if (raw !== undefined) {
        const n = typeof raw === 'number' ? raw : parseFloat(String(raw));
        if (!Number.isNaN(n)) {
      prob = n > 1.001 ? n/100 : n; // treat values >1 as percent
      if (prob < 0) prob = 0; if (prob > 1) prob = 1;
        }
      }
      setItems(prev=>prev.map(it=> it.id===id ? {
        ...it,
        predicted_class: res.data.predicted_class,
        probability: prob,
        probsVector: res.data.raw_probs || undefined,
        latencyMs: latency,
        status:'done'
      }:it));
    } catch(e:any){
      setItems(prev=>prev.map(it=> it.id===id ? { ...it, status:'error', error: e.response?.data?.detail || 'Prediction failed'}:it));
    }
  };

  const onDrop = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); if(e.dataTransfer.files) handleFiles(e.dataTransfer.files); };
  const onDrag = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); if(e.type === 'dragenter' || e.type === 'dragover') setDragActive(true); else if(e.type === 'dragleave') setDragActive(false); };

  const selectForCompare = (id:string) => { setCompare(prev=> prev.includes(id) ? prev.filter(x=>x!==id) : prev.length<2 ? [...prev,id] : prev); };
  const clearHistory = () => { if(confirm('Clear history?')) setItems([]); };

  const classes = ['All','Blurry','Corrupt','Missing_Data','Noisy','Priority'];
  const filteredItems = items.filter(it => activeFilters.length===0 || (it.predicted_class && activeFilters.includes(it.predicted_class)));

  useEffect(()=>{
    const onKey=(e:KeyboardEvent)=>{
      if(e.key.toLowerCase()==='u') fileInputRef.current?.click();
      if(e.key.toLowerCase()==='t') setTheme(t=> t==='dark'?'light':'dark');
      if(e.key.toLowerCase()==='c') setCompare([]);
    };
    window.addEventListener('keydown', onKey);
    return ()=> window.removeEventListener('keydown', onKey);
  },[]);

  return (
    <div className={styles.page}> {/* page wrapper separate from appShell in _app */}
      <Starfield />

  <section className={styles.hero}>
        <div className={styles.heroLeft}>
          <h2 className={styles.heroTitle}>Onboard image quality classification — prioritize high‑value frames</h2>
          <p className={styles.heroSubtitle}>
            CubeSats are constrained by storage, processing, and slow communication. This UI mirrors the development
            pipeline—statistics‑driven preprocessing, a pruned CNN, and TFLite quantization—to rank images for downlink.
          </p>
        </div>
      </section>

      <main className={styles.main}>
        {/* Hidden input available in both states */}
        <input ref={fileInputRef} type="file" accept="image/*" multiple style={{display:'none'}} onChange={e=> e.target.files && handleFiles(e.target.files)} />

        {/* Show big dropzone ONLY before the first image */}
        {items.length === 0 && (
          <section className={dragActive? styles.dropZoneActive: styles.dropZone}
            onDrop={onDrop} onDragEnter={onDrag} onDragOver={onDrag} onDragLeave={onDrag}
            onClick={()=>fileInputRef.current?.click()} aria-label="Upload images via click or drag and drop">
            <p>Drag & Drop or <span className={styles.linkLike}>browse</span> up to 10 images.</p>
            <p className={styles.hint}>Client computes blur/noise metrics; server returns class + confidence.</p>
          </section>
        )}

        <section className={styles.legendRow}>
          <div className={styles.filters}>
            {classes.map(name => {
              const active = name==='All' ? activeFilters.length===0 : activeFilters.includes(name);
              const isClass = name !== 'All';
              const color = isClass ? (CLASS_COLORS as any)[name] : undefined;
              const style = isClass ? { background: color, borderColor: color, color: '#fff', opacity: active ? 1 : 0.6 } : {};
              return (
                <button key={name} className={active? styles.chipActive: styles.chip} style={style as React.CSSProperties}
                  onClick={()=> { if(name==='All') setActiveFilters([]); else setActiveFilters(prev=> prev.includes(name)? prev.filter(c=>c!==name): [...prev,name]); }}>
                  {name}
                </button>
              );
            })}
          </div>
          {/* Legend text removed*/}
          <div className={styles.headerActions}>
            <button className={styles.dangerBtn} onClick={clearHistory}>Clear all</button>
          </div>
        </section>

        <section className={styles.gridTen}> {/* 10 per view where possible */}
          {filteredItems.map(it => {
            const isSelected = compare.includes(it.id);
            const hoverColor = it.predicted_class ? (CLASS_COLORS[it.predicted_class] || '#3b82f6') : 'var(--accent)';
            const outline = it.predicted_class ? (CLASS_COLORS[it.predicted_class] || '#8aa1b4') : 'transparent';
            return (
              <div
                key={it.id}
                className={styles.card}
                data-status={it.status}
                onClick={()=>selectForCompare(it.id)}
                style={{
                  boxShadow: isSelected ? `0 0 0 3px ${outline}` : undefined,
                  ['--hoverColor' as any]: hoverColor
                }}
              >
                <div className={styles.thumbWrap}>
                  <img src={it.objectUrl} alt={it.fileName} />
                  {/* Overlay: standardized pill (top-left) and probability badge (top-right) */}
                  <div className={styles.overlay} aria-hidden={it.status!=='done'}>
                    {it.status==='done' && (
                      <span
                        className={styles.pill}
                        style={{background: CLASS_COLORS[it.predicted_class||''] || '#555'}}
                        title={CLASS_DESCRIPTIONS[it.predicted_class || ''] || it.predicted_class}
                      >
                        <span aria-hidden="true" style={{marginRight:4}}>{CLASS_ICONS[it.predicted_class||''] || ''}</span>
                        {it.predicted_class}
                      </span>
                    )}
                  </div>

                  {/* Card actions (hover): remove/open */}
                  <div className={styles.cardActions} aria-hidden="true">
                    <button className={styles.iconBtn} title="Remove" onClick={(e)=>{ e.stopPropagation(); setItems(prev=>prev.filter(x=>x.id!==it.id)); }}>🗑</button>
                    <a className={styles.iconBtn} title="Open image in new tab" onClick={(e)=> e.stopPropagation()} href={it.objectUrl} target="_blank" rel="noreferrer">↗</a>
                  </div>

                  {it.status==='pending' && <div className={styles.loader} />}
                </div>
                <div className={styles.meta}>
                  <div className={styles.row}>
                    <span className={styles.fileName} title={it.fileName}>{it.fileName}</span>
                  </div>
                  {it.status==='error' && <div className={styles.error}>{it.error}</div>}
                  {it.status==='done' && (
                    <>
                      <div className={styles.predLine}>
                        <div className={styles.classDesc}>{CLASS_DESCRIPTIONS[it.predicted_class || ''] || ''}</div>
                        {it.latencyMs && <span className={styles.latency} title="Model latency">{Math.round(it.latencyMs)} ms</span>}
                      </div>
                      <div className={styles.metricsBlock}>
                        <div className={styles.metricsRow}>
                          <span>Blur var: {it.blurVar? it.blurVar.toFixed(1): '…'}</span>
                          <span>Noise lvl: {it.noiseLvl? it.noiseLvl.toFixed(3): '…'}</span>
                          {typeof it.probability==='number' && (
                            <span>Conf: {(it.probability*100).toFixed(2)}%</span>
                          )}
                        </div>
                        {typeof it.probability==='number' && (
                          <div className={styles.confBarOuter}>
                            <div className={styles.confBarInner} style={{width:`${(it.probability*100).toFixed(1)}%`, background: CLASS_COLORS[it.predicted_class||''] || '#888'}} />
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              </div>
            );
          })}

          {/* After images exist, place a compact add tile at the end */}
          {items.length > 0 && (
            <div
              className={styles.inlineDrop}
              onClick={()=>fileInputRef.current?.click()}
              onDragOver={(e)=>{e.preventDefault();}}
              onDrop={(e)=>{e.preventDefault(); if(e.dataTransfer.files) handleFiles(e.dataTransfer.files);}}
              role="button" aria-label="Add images"
            >
              + Add images <span className={styles.inlineDropHint}>(drag here or click)</span>
            </div>
          )}
        </section>
      </main>

      <footer className={styles.footer}>Items: {items.length} • Tip: use filters to cluster by class</footer>
      <div className={styles.backgroundOrbits} aria-hidden="true"><div/><div/><div/></div>
    </div>
  );
}

// Scientific metrics via canvas (downscaled grayscale); no dependencies
async function computeImageMetrics(url:string): Promise<{blurVar:number, noiseLvl:number}> {
  const img = await loadImage(url);
  const maxW = 256;
  const ratio = img.width ? Math.min(1, maxW / img.width) : 1;
  const w = Math.max(16, Math.round(img.width * ratio));
  const h = Math.max(16, Math.round(img.height * ratio));

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return { blurVar: 0, noiseLvl: 0 };
  }

  ctx.drawImage(img as any, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;

  // Luminance
  const gray = new Float32Array(w * h);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    gray[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  // 3x3 box blur
  const blur = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          s += gray[(y + dy) * w + (x + dx)];
        }
      }
      blur[y * w + x] = s / 9;
    }
  }

  // Residual as noise proxy (std)
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const r = gray[y * w + x] - blur[y * w + x];
      sum += r; sum2 += r * r; n++;
    }
  }
  const mean = sum / n;
  const varr = Math.max(0, (sum2 / n) - mean * mean);
  const noiseLvl = Math.sqrt(varr) / 255;

  // Laplacian variance (edge strength/blur inverse)
  const lapVar = laplacianVariance(gray, w, h);
  return { blurVar: lapVar, noiseLvl };
}

function laplacianVariance(gray: Float32Array, w:number, h:number){
  let sum=0, sum2=0, n=0;
  for(let y=1;y<h-1;y++){
    for(let x=1;x<w-1;x++){
      const c = gray[y*w+x];
      const lap = (-4*c) + gray[y*w + (x-1)] + gray[y*w + (x+1)] + gray[(y-1)*w + x] + gray[(y+1)*w + x];
      sum += lap; sum2 += lap*lap; n++;
    }
  }
  const mean = sum/n; const v = Math.max(0,(sum2/n) - mean*mean);
  return v; // not normalized; used comparatively
}

function loadImage(url:string): Promise<HTMLImageElement> {
  return new Promise((resolve,reject)=>{
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = url;
  });
}

// no extra helpers

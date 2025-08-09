import React, { useEffect, useRef } from 'react';

export default function Starfield() {
  const ref = useRef<HTMLCanvasElement|null>(null);

  useEffect(() => {
    const canvas = ref.current; if (!canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    let w = canvas.width = window.innerWidth;
    let h = canvas.height = window.innerHeight;
    const onResize = () => { w = canvas.width = window.innerWidth; h = canvas.height = window.innerHeight; };
    window.addEventListener('resize', onResize);

    type Star = { x:number; y:number; z:number; s:number; vx:number; vy:number };
    const stars: Star[] = Array.from({ length: 150 }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      z: Math.random() * 0.8 + 0.2,
      s: Math.random() * 1.2 + 0.2,
      vx: (Math.random() - 0.5) * 0.05,
      vy: (Math.random() - 0.5) * 0.05,
    }));

    // Shooting star state
    let shooting: null | { x:number; y:number; vx:number; vy:number; life:number } = null;
    const spawnShooting = () => {
      if (shooting) return;
      const fromTop = Math.random() < 0.5;
      const x = fromTop ? Math.random() * w * 0.6 : -50;
      const y = fromTop ? -20 : Math.random() * h * 0.6;
      const speed = 4 + Math.random() * 3;
      const angle = (Math.PI / 4) + Math.random() * (Math.PI / 6);
      shooting = { x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, life: 60 };
    };
    let shootTimer = 0;

    let raf = 0;
    const draw = () => {
      // Clear (slight trail)
      ctx.fillStyle = 'rgba(0,0,20,0.25)';
      ctx.fillRect(0, 0, w, h);

      // Stars drift + twinkle
      for (const st of stars) {
        st.x += st.vx * (0.5 + st.z);
        st.y += st.vy * (0.5 + st.z);
        if (st.x < -5) st.x = w + 5; if (st.x > w + 5) st.x = -5;
        if (st.y < -5) st.y = h + 5; if (st.y > h + 5) st.y = -5;
        const tw = (Math.sin(performance.now() * 0.003 * st.z) + 1) * 0.25;
        ctx.globalAlpha = 0.25 + tw;
        ctx.fillStyle = 'rgba(255,255,255,0.95)';
        ctx.beginPath();
        ctx.arc(st.x, st.y, st.s, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // Shooting star occasionally
      shootTimer++;
      if (!shooting && shootTimer % 240 === 0) spawnShooting();
      if (shooting) {
        const { x, y } = shooting;
        const grad = ctx.createLinearGradient(x, y, x - shooting.vx * 8, y - shooting.vy * 8);
        grad.addColorStop(0, 'rgba(255,255,255,0.9)');
        grad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.strokeStyle = grad;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(x - shooting.vx * 8, y - shooting.vy * 8);
        ctx.lineTo(x, y);
        ctx.stroke();

        shooting.x += shooting.vx;
        shooting.y += shooting.vy;
        shooting.life -= 1;
        if (shooting.life <= 0 || shooting.x > w + 50 || shooting.y > h + 50) shooting = null;
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', onResize); };
  }, []);

  return <canvas ref={ref} style={{ position: 'fixed', inset: 0, width: '100%', height: '100%', zIndex: -2, pointerEvents: 'none', opacity: 0.25 }} aria-hidden="true" />;
}

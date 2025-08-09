import type { AppProps } from 'next/app';
import Head from 'next/head';
import Link from 'next/link';
import React, { useEffect, useState } from 'react';
import './globals.css';
import styles from '../styles/Home.module.css';

export default function App({ Component, pageProps }: AppProps) {
  const [theme, setTheme] = useState<'dark'|'light'>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('theme');
      if (saved === 'dark' || saved === 'light') return saved;
      const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
      return prefersLight ? 'light' : 'dark';
    }
    return 'dark';
  });

  useEffect(() => { try { localStorage.setItem('theme', theme); } catch {} }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

  return (
    <div className={styles.appShell} data-theme={theme}>
      <Head>
        <title>CubeSat Image Classifier</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <nav className={styles.topNav}>
        <div className={styles.navInner}>
          <span className={styles.brand}>CubeSat Image Classifier</span>
          <div className={styles.navRight}>
            <div className={styles.navLinks}>
              <Link href="/">Home</Link>
              <Link href="/about">About</Link>
              <Link href="/model-card">Model Card</Link>
            </div>
            <button className={styles.themeSwitch} onClick={toggleTheme} title="Toggle theme" aria-label="Toggle theme">
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
      </nav>
      <Component {...pageProps} />
    </div>
  );
}

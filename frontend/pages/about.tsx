import React from 'react';
import styles from '../styles/Home.module.css';

export default function About(){
  return (
    <div className={styles.staticPage}> 
      <h1>Introduction</h1>
      <p>
        Welcome to this solution, which brings together the full pipeline—from data ingestion and preprocessing to
        model training, quantization, and evaluation—into a clear, reproducible story.
      </p>

      <h2>Problem Overview</h2>
      <p>
        CubeSats provide cost‑effective astronomical sensing but face strict limits in size, weight, compute, and
        downlink bandwidth. The VERTECS mission, led by Kyushu Institute of Technology (Kyutech) and partners,
        studies the optical extragalactic background light (EBL). With a small‑aperture telescope and precise
        attitude control, VERTECS captures data that must be prioritized for transmission.
      </p>
      <figure style={{ textAlign: 'center', margin: '1rem 0' }}>
        <img
          src="https://raw.githubusercontent.com/Hack4Dev/CubeSat_ImageClassify/main/pictures/SAT.png"
          alt="VERTECS Satellite Design"
          style={{ maxWidth: '100%', height: 'auto', maxHeight: 420 }}
        />
        <figcaption>Figure: VERTECS Satellite Design (Source: Hack4Dev)</figcaption>
      </figure>

      <p>
        Limited onboard storage, processing capabilities, and slow communication can delay the return of vital data to Earth.
        Onboard machine learning helps the spacecraft pre‑select the most valuable images for downlink, maximizing the
        science return under constraints.
      </p>

      <h2>Objective</h2>
      <p>
        Develop a lightweight, efficient image classifier that accurately categorizes CubeSat imagery so high‑priority
        frames are identified for downlink under strict resource constraints. The approach balances computational
        efficiency with accuracy for resource‑limited platforms.
      </p>

      <h2>Five Quality Classes</h2>
      <ul className={styles.list}> 
        <li><strong>Blurry</strong>: Motion/defocus induced loss of detail.</li>
        <li><strong>Corrupt</strong>: Sensor/transmission artifacts and stray light effects.</li>
        <li><strong>Missing_Data</strong>: Partial frame loss or blank segments.</li>
        <li><strong>Noisy</strong>: Radiation/electronic noise dominated frames.</li>
        <li><strong>Priority</strong>: Clear scientifically valuable imagery to downlink first.</li>
      </ul>
      <h2>Pipeline Summary</h2>
      <p>
        The development workflow integrates global statistics normalization, model pruning, TFLite quantization, and
        evaluation in CPU‑constrained settings. This frontend adds interactive exploration and comparison to help
        contextualize predictions and quality metrics.
      </p>
      <p className={styles.smallNote}>Frontend‑only changes — model and backend deployment remain unchanged.</p>
    </div>
  );
}

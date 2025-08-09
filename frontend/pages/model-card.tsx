import React from 'react';
import styles from '../styles/Home.module.css';

export default function ModelCard(){
  return (
    <div className={styles.staticPage}>
      <h1>Model Card</h1>
      <figure className={styles.figure}>
        <img
          src="https://raw.githubusercontent.com/Hack4Dev/CubeSat_ImageClassify/main/pictures/SAT.png"
          alt="VERTECS Satellite Design"
        />
        <figcaption>VERTECS Satellite Design (Source: Hack4Dev)</figcaption>
      </figure>
      <section>
        <h2>Overview</h2>
        <p>
          Lightweight CNN (pruned & quantized to TFLite) for 5-way quality classification in a CubeSat context:
          Blurry, Corrupt, Missing_Data, Noisy, Priority.
        </p>
      </section>
      <section>
        <h2>Intended Use</h2>
        <p>
          Onboard screening and prioritization of imagery before downlink. Not a substitute for final scientific
          quality vetting. Designed for low-latency CPU execution.
        </p>
      </section>
      <section>
        <h2>Problem Overview</h2>
        <p>
          CubeSats are effective but resource-constrained. Onboard inference helps prioritize high-value images under
          downlink and compute limits. This card consolidates the context so the home page can stay task-focused.
        </p>
      </section>
      <section>
        <h2>Data Summary</h2>
        <p>
          Training/validation/test splits derived from preprocessed NPY tensors (9,711 / 3,237 / 3,237). Class
          imbalance present (Priority most frequent; Corrupt & Missing_Data minority).
        </p>
      </section>
      <section>
        <h2>Preprocessing</h2>
        <ul className={styles.list}>
          <li>Global mean/variance statistics computed over training images.</li>
          <li>Resize to input resolution; bilinear interpolation.</li>
          <li>Normalization (/255 and statistical normalization as appropriate).</li>
        </ul>
      </section>
      <section>
        <h2>Model Efficiency</h2>
        <ul className={styles.list}>
          <li>Pruning schedule during training (progressive sparsity).</li>
          <li>Post-training quantization to TFLite.</li>
          <li>Mixed precision for faster convergence (training).</li>
        </ul>
      </section>
      <section>
        <h2>Limitations</h2>
        <ul className={styles.list}>
          <li>Class imbalance can reduce minority recall.</li>
          <li>Quantization may affect calibration.</li>
          <li>Distribution shift vs. real in-orbit conditions is possible.</li>
        </ul>
      </section>
      <p className={styles.smallNote}>Summarizes notebook concepts without changing backend behavior.</p>
    </div>
  );
}

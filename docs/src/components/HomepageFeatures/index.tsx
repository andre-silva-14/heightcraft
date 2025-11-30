import type { ReactNode } from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  image: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'AI Upscaling',
    image: require('@site/static/img/feature-ai.png').default,
    description: (
      <>
        Transform low-resolution inputs into crisp, high-detail heightmaps.
        Increase bit depth (8-bit → 16/32-bit) with AI hallucination for
        game-ready terrains.
      </>
    ),
  },
  {
    title: 'LiDAR Streaming',
    image: require('@site/static/img/feature-lidar.png').default,
    description: (
      <>
        Process massive <code>.las</code> and <code>.laz</code> point clouds
        efficiently. Heightcraft uses a streaming pipeline to handle gigabytes
        of data with minimal memory.
      </>
    ),
  },
  {
    title: 'Mesh Baking',
    image: require('@site/static/img/feature-mesh.png').default,
    description: (
      <>
        Convert 3D meshes (<code>.obj</code>, <code>.stl</code>) into
        heightmaps. Perfect for baking terrain geometry from modeling software
        like Blender or ZBrush.
      </>
    ),
  },
];

function Feature({ title, image, description }: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img src={image} className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

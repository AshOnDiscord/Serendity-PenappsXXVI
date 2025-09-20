// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator } from "@uwdata/mosaic-core";

export interface DatasetConfig {
  numPoints: number;
  numCategories: number;
  numSubClusters: number;
}

export interface Row {
  identifier: number;
  x: number;
  y: number;
  category: number;
  text: string;
}

export function generateSampleDataset(config: DatasetConfig): Row[] {
  let rng = RNG(42);
  let clusters: {
    x: number;
    y: number;
    sigmaX: number;
    sigmaY: number;
    angle: number;
    category: number;
    text: string;
  }[] = [];
  for (let c = 0; c < config.numCategories; c++) {
    for (let j = 0; j < config.numSubClusters; j++) {
      clusters.push({
        x: randomUniform(-10, 10, rng),
        y: randomUniform(-10, 10, rng),
        sigmaX: randomUniform(0.2, 2, rng),
        sigmaY: randomUniform(0.2, 2, rng),
        angle: rng() * Math.PI * 2,
        category: c,
        text: randomChoices(words, 3, rng).join(", "),
      });
    }
  }
  let result: Row[] = [];
  for (let i = 0; i < config.numPoints; i++) {
    let cluster = clusters[Math.floor(rng() * clusters.length)];
    let x = randomNormal(cluster.x, cluster.sigmaX, rng);
    let y = randomNormal(cluster.y, cluster.sigmaY, rng);
    result.push({
      identifier: i,
      x: Math.cos(cluster.angle) * x - Math.sin(cluster.angle) * y,
      y: Math.sin(cluster.angle) * x + Math.cos(cluster.angle) * y,
      category: cluster.category,
      text: cluster.text + ` (${i})`,
    });
  }
  return result;
}

export async function createSampleDataTable(coordinator: Coordinator, table: string, count: number) {
  await coordinator.exec(`
    SELECT setseed(0.42);

    CREATE OR REPLACE MACRO random_normal() AS (sqrt(-2 * LN(random())) * cos(2 * PI() * random()));
    CREATE OR REPLACE MACRO random_choice(list, f) AS list_element(list, 1 + floor(pow(random(), f) * len(list))::INT);
    CREATE OR REPLACE MACRO random_invalid(x) AS CASE WHEN random() < 0.02 THEN random_choice(['NaN','Inf','-Inf', Null], 2)::DOUBLE ELSE x END;
    CREATE OR REPLACE MACRO shuffle_hash(x, v) AS x + hash(v) % 5;
    CREATE OR REPLACE MACRO random_concat(list, n) AS list_reduce(list_transform(range(n), x -> random_choice(list, 1)), (a, b) -> concat(a, ', ', b));;

    CREATE OR REPLACE TABLE ${table} AS (
      SELECT
        id,
        random_concat([${words.map((x) => `'${x}'`).join(",")}], 8) AS text,
        random_choice([${words.map((x) => `'${x}'`).join(",")}, null], 2) AS var_category,
        concat('Category', (150 * abs(sin(20 * pow(random(), 5))))::INT::TEXT) AS var_many_category,
        random() AS var_uniform,
        random_invalid(random_normal()) AS var_normal,
        random_invalid(exp(random_normal() + 1)) AS var_log_normal,
        random_invalid(random_normal() + 1000) AS var_normal_biased,
        random_invalid(random_normal() - 1000) AS var_normal_biased_negative,
        random_normal() AS x,
        random_normal() AS y
      FROM range(0, ${count}) t(id)
    );
    UPDATE ${table} SET var_normal = var_normal + 5 WHERE var_log_normal < exp(random());
    UPDATE ${table} SET var_normal_biased = var_normal_biased + random() * var_normal;
    UPDATE ${table} SET var_normal_biased_negative = var_normal_biased_negative + random() * var_normal;
    UPDATE ${table} SET
      x = sin(10 * shuffle_hash(var_uniform, var_category)) * x + cos(5 * shuffle_hash(var_uniform, var_category)) * y,
      y = sin(4 * shuffle_hash(var_uniform, var_category)) * x + cos(18 * shuffle_hash(var_uniform, var_category)) * y;
    UPDATE ${table} SET x = x + 5 * fmod(floor(var_uniform * 24 + random()), 5);
    UPDATE ${table} SET y = y + 5 * floor(floor(var_uniform * 24 + random()) / 5);
  `);
}

export class SplitMix64 {
  seed: bigint;

  constructor(seed: number) {
    this.seed = BigInt(seed);
  }

  next() {
    let z = (this.seed += BigInt("0x9E3779B97F4A7C15"));
    z = (z ^ (z >> BigInt(30))) * BigInt("0xBF58476D1CE4E5B9");
    z = (z ^ (z >> BigInt(27))) * BigInt("0x94D049BB133111EB");
    return z ^ (z >> BigInt(31));
  }

  nextNumber() {
    return Number(this.next() & BigInt("0xFFFFFFFFFFFFFFFF"));
  }
}

export function RNG(seed: number) {
  let algo = new SplitMix64(seed);
  return () => {
    return algo.nextNumber() / 0xffffffffffffffff;
  };
}

export function randomUniform(a: number, b: number, rng: () => number) {
  return a + (b - a) * rng();
}

export function randomNormal(mean: number, stdev: number, rng: () => number) {
  const u = 1 - rng();
  const v = rng();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdev + mean;
}

export function randomChoices(list: string[], count: number, rng: () => number) {
  let result: string[] = [];
  for (let i = 0; i < count; i++) {
    result.push(list[Math.floor(rng() * list.length)]);
  }
  return result;
}

let words = [
  "Gala",
  "Fuji",
  "Granny Smith",
  "Honeycrisp",
  "Red Delicious",
  "Golden Delicious",
  "Braeburn",
  "Pink Lady",
  "Cortland",
  "Jonagold",
  "Empire",
  "Ambrosia",
  "Jazz",
  "Cameo",
  "Envy",
  "Mutsu",
  "Opal",
  "Winesap",
];

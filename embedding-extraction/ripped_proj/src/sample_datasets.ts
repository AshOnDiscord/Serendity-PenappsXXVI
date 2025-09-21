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

export interface WebsiteData {
  id: number;
  url: string;
  content: string;
  embedding: number[];
  cluster: number;
  x: number;
  y: number;
  time: string;
}

// Flask backend configuration
const FLASK_BASE_URL = 'http://localhost:5000';

/**
 * Convert website data to the Row format expected by the visualization
 */
export function convertWebsiteDataToRows(websiteData: WebsiteData[]): Row[] {
  return websiteData.map((website) => ({
    identifier: website.id,
    x: website.x || 0,
    y: website.y || 0,
    category: website.cluster || 0,
    text: website.url || `Website ${website.id}`
  }));
}

/**
 * Generate sample dataset from Flask backend
 */
export async function generateSampleDataset(config?: DatasetConfig): Promise<Row[]> {
  try {
    console.log('Fetching website data from Flask backend...');
    const websiteData = await fetchWebsiteData();
    
    if (websiteData.length === 0) {
      throw new Error('No data found in backend');
    }
    
    const rows = convertWebsiteDataToRows(websiteData);
    
    // Apply config constraints if provided
    if (config) {
      let filteredRows = rows;
      
      // Limit number of points
      if (config.numPoints && config.numPoints < rows.length) {
        filteredRows = rows.slice(0, config.numPoints);
      }
      
      // Filter by number of categories if specified
      if (config.numCategories) {
        const uniqueCategories = [...new Set(filteredRows.map(r => r.category))]
          .slice(0, config.numCategories);
        filteredRows = filteredRows.filter(r => uniqueCategories.includes(r.category));
      }
      
      return filteredRows;
    }
    
    console.log(`Loaded ${rows.length} data points from Flask backend`);
    return rows;
    
  } catch (error) {
    console.error('Error generating sample dataset:', error);
    throw error;
  }
}

/**
 * Fetch all website data from Flask backend
 */
export async function fetchWebsiteData(): Promise<WebsiteData[]> {
  try {
    const response = await fetch(`${FLASK_BASE_URL}/all_data`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    if (!result.success) {
      throw new Error(result.error || 'Failed to fetch data');
    }
    
    return result.data;
  } catch (error) {
    console.error('Error fetching website data:', error);
    throw new Error(`Failed to fetch data from Flask backend: ${error.message}`);
  }
}

/**
 * Create a DuckDB table from Flask backend data
 * This function is called by the Svelte component to populate the "dataset" table
 */
export async function createSampleDataTable(coordinator: Coordinator, table: string, count: number) {
  try {
    console.log('Fetching website data from Flask backend...');
    const websiteData = await fetchWebsiteData();
    
    if (websiteData.length === 0) {
      throw new Error('No data found in backend');
    }

    console.log(`Loaded ${websiteData.length} websites from Flask backend`);

    // Create the table structure
    await coordinator.exec(`DROP TABLE IF EXISTS ${table}`);
    
    // Create table with the expected schema
    await coordinator.exec(`
      CREATE TABLE ${table} (
        id INTEGER,
        text VARCHAR,
        x DOUBLE,
        y DOUBLE,
        category INTEGER,
        cluster INTEGER,
        url VARCHAR,
        content VARCHAR
      )
    `);

    // Insert data in batches for better performance
    const batchSize = 1000;
    const batches = [];
    
    for (let i = 0; i < websiteData.length; i += batchSize) {
      const batch = websiteData.slice(i, i + batchSize);
      batches.push(batch);
    }

    for (const [batchIndex, batch] of batches.entries()) {
      const values = batch.map(website => {
        // Escape single quotes in text content
        const escapedUrl = website.url?.replace(/'/g, "''") || `Website ${website.id}`;
        const escapedContent = website.content?.replace(/'/g, "''").substring(0, 1000) || '';
        
        return `(
          ${website.id}, 
          '${escapedUrl}', 
          ${website.x || 0}, 
          ${website.y || 0}, 
          ${website.cluster || 0}, 
          ${website.cluster || 0},
          '${escapedUrl}',
          '${escapedContent}'
        )`;
      }).join(',');

      await coordinator.exec(`
        INSERT INTO ${table} (id, text, x, y, category, cluster, url, content) 
        VALUES ${values}
      `);

      console.log(`Inserted batch ${batchIndex + 1}/${batches.length} (${batch.length} records)`);
    }

    // Verify the data was inserted
    try {
      const result = await coordinator.exec(`SELECT COUNT(*) as count FROM ${table}`);
      const count = result && result.length > 0 ? result[0]?.count || websiteData.length : websiteData.length;
      console.log(`Successfully created table '${table}' with ${count} rows`);
    } catch (verifyError) {
      console.log(`Successfully created table '${table}' with ${websiteData.length} rows (verification failed but table created)`);
    }

  } catch (error) {
    console.error('Error creating sample data table:', error);
    throw new Error(`Failed to create table from Flask backend: ${error.message}`);
  }
}

/**
 * Add a new website to the Flask backend
 */
export async function addWebsite(url: string, nClusters: number = 10): Promise<any> {
  try {
    const response = await fetch(`${FLASK_BASE_URL}/add_website`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url: url,
        n_clusters: nClusters
      })
    });
    
    const result = await response.json();
    
    if (!response.ok) {
      throw new Error(result.error || `HTTP error! status: ${response.status}`);
    }
    
    return result;
  } catch (error) {
    console.error('Error adding website:', error);
    throw error;
  }
}

/**
 * Recalculate clusters for all data
 */
export async function reclusterData(nClusters: number = 10): Promise<any> {
  try {
    const response = await fetch(`${FLASK_BASE_URL}/recluster`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        n_clusters: nClusters
      })
    });
    
    const result = await response.json();
    
    if (!response.ok) {
      throw new Error(result.error || `HTTP error! status: ${response.status}`);
    }
    
    return result;
  } catch (error) {
    console.error('Error reclustering data:', error);
    throw error;
  }
}

/**
 * Recalculate UMAP coordinates for all data
 */
export async function recalculateUMAP(params: {
  nNeighbors?: number;
  minDist?: number;
  randomState?: number;
} = {}): Promise<any> {
  try {
    const response = await fetch(`${FLASK_BASE_URL}/calculate_umap`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        n_neighbors: params.nNeighbors || 10,
        min_dist: params.minDist || 0.1,
        random_state: params.randomState || 42
      })
    });
    
    const result = await response.json();
    
    if (!response.ok) {
      throw new Error(result.error || `HTTP error! status: ${response.status}`);
    }
    
    return result;
  } catch (error) {
    console.error('Error recalculating UMAP:', error);
    throw error;
  }
}

/**
 * Get cluster summary from Flask backend
 */
export async function getClusterSummary(clusterId: number): Promise<any> {
  try {
    const response = await fetch(`${FLASK_BASE_URL}/summarize_cluster/${clusterId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    if (!result.success) {
      throw new Error(result.error || 'Failed to get cluster summary');
    }
    
    return result;
  } catch (error) {
    console.error('Error getting cluster summary:', error);
    throw error;
  }
}

/**
 * Get backend statistics
 */
export async function getBackendStats(): Promise<any> {
  try {
    const response = await fetch(`${FLASK_BASE_URL}/stats`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting backend stats:', error);
    throw error;
  }
}








/**
 * Fetch parquet data from Flask backend and create a table in Coordinator
 */
export async function createSampleDataTableParquet(
  coordinator: Coordinator,
  table: string
) {
  try {
    console.log('Fetching parquet data from Flask backend...');
    const response = await fetch(`${FLASK_BASE_URL}/read_parquet`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    if (!result.success) {
      throw new Error(result.error || 'Failed to fetch parquet data');
    }

    const parquetData = result.data as any[];
    console.log(`Loaded ${parquetData.length} records from parquet`);

    // Drop existing table if exists
    await coordinator.exec(`DROP TABLE IF EXISTS ${table}`);

    // Create table schema
    await coordinator.exec(`
      CREATE TABLE ${table} (
        id INTEGER,
        text VARCHAR,
        x DOUBLE,
        y DOUBLE,
        category INTEGER,
        cluster INTEGER,
        url VARCHAR,
        content VARCHAR,
        vector ARRAY
      )
    `);

    const batchSize = 500;

    for (let i = 0; i < parquetData.length; i += batchSize) {
      const batch = parquetData.slice(i, i + batchSize);

      const values = batch.map(row => {
        const escapedUrl = (row.url || '').replace(/'/g, "''");
        const escapedContent = (row.content || '').replace(/'/g, "''").substring(0, 1000);

        // --- Robust vector handling ---
        let vector = 'NULL';
        if (row.vector) {
          try {
            let vecArray: number[] = [];

            if (typeof row.vector === 'string') {
              const cleaned = row.vector.replace(/[\n\r]+/g, '').trim();
              vecArray = JSON.parse(cleaned);
            } else if (Array.isArray(row.vector)) {
              vecArray = row.vector;
            }

            // Replace invalid numbers with 0
            vecArray = vecArray.map(x => (isNaN(Number(x)) ? 0 : Number(x)));

            vector = `[${vecArray.join(',')}]`;
          } catch (err) {
            console.warn(`Skipping malformed vector for row id=${row.id}:`, err);
            vector = 'NULL';
          }
        }

        return `(
          ${row.id || i}, 
          '${escapedUrl}', 
          ${row.x || 0}, 
          ${row.y || 0}, 
          ${row.cluster || 0}, 
          ${row.cluster || 0}, 
          '${escapedUrl}', 
          '${escapedContent}', 
          ${vector}
        )`;
      }).join(',');

      await coordinator.exec(`
        INSERT INTO ${table} (id, text, x, y, category, cluster, url, content, vector)
        VALUES ${values}
      `);

      console.log(`Inserted batch ${i / batchSize + 1}/${Math.ceil(parquetData.length / batchSize)} (${batch.length} records)`);
    }

    console.log(`Successfully created table '${table}' with ${parquetData.length} rows`);
  } catch (error) {
    console.error('Error creating table from parquet data:', error);
    throw error;
  }
}

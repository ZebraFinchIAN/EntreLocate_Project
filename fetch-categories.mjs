import fs from "fs/promises";
import fetch from "node-fetch";

async function fetchBusinessCategories() {
  const query = `
    [out:json][timeout:25];
    area["name"="İstanbul"]["admin_level"="4"]->.searchArea;
    (
      node(area.searchArea)[shop];
      node(area.searchArea)[amenity];
      node(area.searchArea)[office];
      node(area.searchArea)[craft];
    );
    out body;
  `;

  const response = await fetch("https://overpass-api.de/api/interpreter", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded"
    },
    body: `data=${encodeURIComponent(query)}`
  });

  if (!response.ok) {
    throw new Error(`Overpass API request failed: ${response.status}`);
  }

  const data = await response.json();

  const categories = new Set();

  for (const el of data.elements) {
    if (el.tags) {
      ["shop", "amenity", "office", "craft"].forEach(key => {
        if (el.tags[key]) {
          categories.add(el.tags[key]);
        }
      });
    }
  }

  const categoryList = Array.from(categories).sort();
  await fs.writeFile("categories.json", JSON.stringify(categoryList, null, 2));
  console.log(`Saved ${categoryList.length} categories to categories.json`);
}

fetchBusinessCategories().catch(console.error);

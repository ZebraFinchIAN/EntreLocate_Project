import fs from "fs/promises";
import fetch from "node-fetch";

async function fetchCities() {
  const query = `
    [out:json][timeout:25];
    area["ISO3166-1"="TR"][admin_level=2]->.searchArea;
    node["place"="city"](area.searchArea);
    out body;
  `;

  const response = await fetch("https://overpass-api.de/api/interpreter", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded"
    },
    body: `data=${encodeURIComponent(query)}`
  });

  const data = await response.json();

  const cities = data.elements
    .filter(el => el.tags && el.tags.name)
    .map(el => el.tags.name)
    .sort();

  await fs.writeFile("cities.json", JSON.stringify(cities, null, 2));
  console.log(`Saved ${cities.length} cities to cities.json`);
}

fetchCities();
